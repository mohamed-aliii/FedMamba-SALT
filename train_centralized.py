#!/usr/bin/env python
"""
train_centralized.py -- Centralized SALT pre-training for FedMamba-SALT.

Trains a lightweight Inception-Mamba student encoder to match the embedding
space of a frozen MAE-pretrained ViT-B/16 teacher.  The learning signal comes
from asymmetric augmentation: the teacher sees a clean view while the student
sees a heavily corrupted view of the same image.

Usage:
    python train_centralized.py --data_path /path/to/imagefolder --epochs 100

Expected training behavior (healthy run)
=========================================
  Loss = SmoothL1(student_proj, teacher_emb)  (range [0, +inf))
  Epoch   1:  loss ~ 0.3-0.5  (random init, large vector mismatch)
  Epoch  10:  loss ~ 0.1-0.3  (warmup complete, student converging)
  Epoch  50:  loss ~ 0.01-0.1
  Epoch 100:  loss ~ 0.005-0.05  (plateau)

  student_std should track towards teacher_std (~0.04).
  If student_std diverges wildly (>10x teacher_std) or drops
  to near zero (<0.2x teacher_std), investigate.
=========================================
"""

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from augmentations.medical_aug import (
    DualViewDataset, get_teacher_transform, get_student_transform,
)
from augmentations.retina_dataset import RetinaDataset
from models.inception_mamba import InceptionMambaEncoder
from models.vit_teacher import FrozenViTTeacher
from objectives.salt_loss import ProjectionHead, embedding_std, salt_loss
from utils.ckpt_compat import safe_torch_load

# ======================================================================
# Constants
# ======================================================================
WARMUP_EPOCHS = 10
COLLAPSE_RATIO = 0.2  # warn if student_std < teacher_std * this ratio
METRICS_FILENAME = "training_metrics.csv"


# ======================================================================
# GPU memory tracking
# ======================================================================
def get_gpu_memory_mb() -> dict:
    """
    Return current and peak GPU memory usage in MB.
    Returns zeros if CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return {"gpu_mem_allocated_mb": 0.0, "gpu_mem_reserved_mb": 0.0, "gpu_mem_peak_mb": 0.0}
    return {
        "gpu_mem_allocated_mb": torch.cuda.memory_allocated() / (1024 ** 2),
        "gpu_mem_reserved_mb": torch.cuda.memory_reserved() / (1024 ** 2),
        "gpu_mem_peak_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
    }


# ======================================================================
# CSV Metrics Logger
# ======================================================================
class MetricsLogger:
    """
    Append-mode CSV logger that writes one row per epoch.

    Columns: epoch, loss, student_std, teacher_std, lr, epoch_time_s,
             gpu_mem_allocated_mb, gpu_mem_reserved_mb, gpu_mem_peak_mb
    """

    COLUMNS = [
        "epoch", "loss", "student_std", "teacher_std", "lr",
        "epoch_time_s", "gpu_mem_allocated_mb", "gpu_mem_reserved_mb",
        "gpu_mem_peak_mb",
    ]

    def __init__(self, output_dir: str):
        self.path = os.path.join(output_dir, METRICS_FILENAME)
        # Write header only if file doesn't exist (supports resume)
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.COLUMNS)

    def log(self, epoch: int, loss: float, student_std: float,
            teacher_std: float, lr: float, epoch_time: float,
            gpu_mem: dict) -> None:
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                f"{loss:.6f}",
                f"{student_std:.6f}",
                f"{teacher_std:.6f}",
                f"{lr:.2e}",
                f"{epoch_time:.1f}",
                f"{gpu_mem['gpu_mem_allocated_mb']:.1f}",
                f"{gpu_mem['gpu_mem_reserved_mb']:.1f}",
                f"{gpu_mem['gpu_mem_peak_mb']:.1f}",
            ])


# ======================================================================
# YAML config loading
# ======================================================================
def load_yaml_config(path: str) -> dict:
    """
    Load a YAML config file and return a dict of key-value pairs.

    Only keys that are valid argparse argument names are returned.
    Unknown keys are silently skipped (they may be eval-only settings).
    """
    import yaml

    try:
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"[ERROR] Config file not found: {path}")
        raise

    if not isinstance(raw, dict):
        raise ValueError(
            f"Expected YAML config to be a mapping (dict), got {type(raw).__name__}"
        )

    return raw


# ======================================================================
# CLI
# ======================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FedMamba-SALT: Centralized self-supervised pre-training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file (parsed first via two-pass approach)
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a YAML config file. Values from the file are used as "
             "defaults; any CLI flags override them.",
    )

    # Data
    parser.add_argument(
        "--data_path", type=str, default=None,
        help="Root of an ImageFolder dataset (must contain a train/ subdirectory)",
    )
    parser.add_argument(
        "--teacher_ckpt", type=str, default="data/ckpts/mae_vit_base.pth",
        help="Path to the MAE ViT-B/16 teacher checkpoint",
    )

    # Output
    parser.add_argument(
        "--output_dir", type=str, default="outputs",
        help="Directory to save training checkpoints",
    )

    # Training hyper-parameters
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch_size",   type=int,   default=128)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument("--save_every",   type=int,   default=10)
    parser.add_argument("--device",       type=str,   default="cuda")

    # ------------------------------------------------------------------
    # Two-pass parsing: extract --config first, load YAML, set defaults
    # ------------------------------------------------------------------
    known, _ = parser.parse_known_args()

    if known.config is not None:
        yaml_dict = load_yaml_config(known.config)

        # Filter to only keys that match valid argparse destinations
        valid_keys = {a.dest for a in parser._actions}
        filtered = {k: v for k, v in yaml_dict.items() if k in valid_keys}
        skipped = {k for k in yaml_dict if k not in valid_keys}

        parser.set_defaults(**filtered)

        if skipped:
            print(f"[Config] Skipped non-training keys: {sorted(skipped)}")

    # Full parse (CLI flags override YAML defaults)
    args = parser.parse_args()

    # Validate that data_path was provided (either via CLI or YAML)
    if args.data_path is None:
        parser.error("--data_path is required (via CLI or YAML config)")

    # Log loaded config
    if args.config is not None:
        print(f"[Config] Loaded from: {args.config}")
        yaml_dict = load_yaml_config(args.config)
        valid_keys = {a.dest for a in parser._actions}
        for k, v in sorted(yaml_dict.items()):
            if k in valid_keys:
                actual = getattr(args, k, v)
                override = " (overridden by CLI)" if actual != v else ""
                print(f"  {k}: {v}{override}")

    return args


# ======================================================================
# Data
# ======================================================================
def build_dataloader(args: argparse.Namespace) -> DataLoader:
    """Build the DualViewDataset + DataLoader from a RetinaDataset (SSL-FL format)."""
    # Load the Retina dataset in SSL-FL format
    base_ds = RetinaDataset(
        data_path=args.data_path,
        phase="train",
        split_type="central",
        split_csv="train.csv",
    )

    # Wrap with dual-view transforms using Retina normalization
    dual_ds = DualViewDataset(
        base_ds,
        teacher_transform=get_teacher_transform(dataset="retina"),
        student_transform=get_student_transform(dataset="retina"),
    )

    loader = DataLoader(
        dual_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    print(f"Dataset: {len(base_ds)} images from {args.data_path}")
    print(f"DataLoader: {len(loader)} batches of {args.batch_size}")
    return loader


# ======================================================================
# Models
# ======================================================================
def build_models(args: argparse.Namespace):
    """Instantiate teacher, student, and projection head."""

    # Teacher: frozen ViT-B/16 (checkpoint is required)
    if not os.path.isfile(args.teacher_ckpt):
        raise FileNotFoundError(
            f"Teacher checkpoint not found: {args.teacher_ckpt}\n"
            f"Download the MAE ViT-B/16 checkpoint and place it at this path.\n"
            f"Training with random teacher weights produces meaningless results."
        )
    teacher = FrozenViTTeacher(ckpt_path=args.teacher_ckpt).to(args.device)

    # Student: Inception-Mamba encoder
    student = InceptionMambaEncoder(
        patch_size=16, embed_dim=256, depth=4, out_dim=768,
    ).to(args.device)

    # Projection head: BYOL-style MLP
    projector = ProjectionHead(
        in_dim=768, hidden_dim=2048, out_dim=768,
    ).to(args.device)

    # -------------------------------------------------------------------
    # Verify teacher is fully frozen
    # -------------------------------------------------------------------
    teacher_trainable = sum(
        p.numel() for p in teacher.parameters() if p.requires_grad
    )
    assert teacher_trainable == 0, (
        f"Teacher has {teacher_trainable} trainable params -- should be 0!"
    )

    # -------------------------------------------------------------------
    # Print trainable parameter counts
    # -------------------------------------------------------------------
    student_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    proj_params = sum(p.numel() for p in projector.parameters() if p.requires_grad)
    total_params = student_params + proj_params

    print(f"\n{'='*55}")
    print(f"  Teacher (frozen): {sum(p.numel() for p in teacher.parameters()) / 1e6:.2f}M params")
    print(f"  Student encoder:  {student_params / 1e6:.2f}M trainable params")
    print(f"  Projection head:  {proj_params / 1e6:.2f}M trainable params")
    print(f"  Total trainable:  {total_params / 1e6:.2f}M params")
    print(f"{'='*55}\n")

    return teacher, student, projector


# ======================================================================
# Optimizer & Scheduler
# ======================================================================
def build_optimizer_and_scheduler(
    student: nn.Module,
    projector: nn.Module,
    args: argparse.Namespace,
):
    """
    AdamW on student + projector params only (teacher excluded).
    LR schedule: 10-epoch linear warmup (lr/10 -> lr) followed by
    cosine annealing over the remaining epochs.
    """
    params = list(student.parameters()) + list(projector.parameters())

    optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # Linear warmup: start_factor=0.1 means lr starts at lr*0.1
    warmup = LinearLR(
        optimizer,
        start_factor=1.0 / 10.0,
        total_iters=WARMUP_EPOCHS,
    )
    # Cosine annealing over the remaining epochs
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs - WARMUP_EPOCHS),
    )
    # Compose: warmup for first WARMUP_EPOCHS, then cosine
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[WARMUP_EPOCHS],
    )

    return optimizer, scheduler


# ======================================================================
# Checkpointing
# ======================================================================
def save_checkpoint(
    student: nn.Module,
    projector: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    loss: float,
    output_dir: str,
    name: str,
) -> None:
    """Save a training checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    torch.save(
        {
            "epoch": epoch,
            "loss": loss,
            "student_state_dict": student.state_dict(),
            "projector_state_dict": projector.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        path,
    )


def try_resume(
    output_dir: str,
    student: nn.Module,
    projector: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
) -> int:
    """
    If ckpt_latest.pth exists in output_dir, resume from it.
    Returns the epoch to start from (0 if no checkpoint found).
    """
    latest_path = os.path.join(output_dir, "ckpt_latest.pth")
    if not os.path.isfile(latest_path):
        return 0

    print(f"[RESUME] Loading checkpoint: {latest_path}")
    ckpt = safe_torch_load(latest_path, map_location=device)

    student.load_state_dict(ckpt["student_state_dict"])
    projector.load_state_dict(ckpt["projector_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    start_epoch = ckpt["epoch"] + 1  # resume from the NEXT epoch
    prev_loss = ckpt["loss"]
    print(f"[RESUME] Resuming from epoch {start_epoch} (prev loss: {prev_loss:.4f})")
    return start_epoch


# ======================================================================
# Training loop (one epoch)
# ======================================================================
def train_one_epoch(
    teacher: nn.Module,
    student: nn.Module,
    projector: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> tuple:
    """
    Run one epoch of SALT training.
    Returns (avg_loss, avg_align, avg_var, avg_student_std, avg_teacher_std).
    """
    student.train()
    projector.train()
    # teacher stays in eval() permanently (enforced inside its forward())

    total_loss = 0.0
    total_align = 0.0
    total_var = 0.0
    total_s_std = 0.0
    total_t_std = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc="  Batches", leave=False, ncols=90)
    for teacher_view, student_view in pbar:
        teacher_view = teacher_view.to(device, non_blocking=True)
        student_view = student_view.to(device, non_blocking=True)

        # ----- Teacher embedding (frozen, no gradient) -----
        with torch.no_grad():
            t_emb = teacher(teacher_view)               # (B, 768)

        # ----- Student embedding + projection -----
        s_emb = student(student_view)                    # (B, 768)
        s_proj = projector(s_emb)                        # (B, 768)

        # ----- SALT loss (Smooth L1 direct manifold distillation) -----
        loss, align_loss, var_loss = salt_loss(s_proj, t_emb)

        # ----- Collapse diagnostics -----
        s_std = embedding_std(s_proj.detach())
        t_std = embedding_std(t_emb.detach())

        # ----- Backward + gradient clipping + step -----
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(student.parameters()) + list(projector.parameters()),
            max_norm=1.0,
        )
        optimizer.step()

        # ----- Accumulate metrics -----
        total_loss += loss.item()
        total_align += align_loss.item()
        total_var += var_loss.item()
        total_s_std += s_std
        total_t_std += t_std
        n_batches += 1

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            align=f"{align_loss.item():.4f}",
            var=f"{var_loss.item():.3f}",
            s_std=f"{s_std:.3f}",
        )

    avg_loss = total_loss / max(1, n_batches)
    avg_align = total_align / max(1, n_batches)
    avg_var = total_var / max(1, n_batches)
    avg_s_std = total_s_std / max(1, n_batches)
    avg_t_std = total_t_std / max(1, n_batches)

    return avg_loss, avg_align, avg_var, avg_s_std, avg_t_std


# ======================================================================
# Main
# ======================================================================
def main() -> None:
    args = parse_args()

    print("=" * 55)
    print("  FedMamba-SALT: Centralized Pre-training")
    print("=" * 55)
    print(f"  Device:     {args.device}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR:         {args.lr}")
    print(f"  Warmup:     {WARMUP_EPOCHS} epochs (lr/{10} -> lr)")
    print(f"  Output:     {args.output_dir}")
    print()

    # ----- Build components -----
    dataloader = build_dataloader(args)
    teacher, student, projector = build_models(args)
    optimizer, scheduler = build_optimizer_and_scheduler(student, projector, args)

    # ----- Resume from checkpoint if available -----
    start_epoch = try_resume(
        args.output_dir, student, projector, optimizer, scheduler, args.device,
    )

    # ----- Training loop -----
    # Expected loss trajectory (healthy training, Smooth L1):
    #   Loss = SmoothL1(student_proj, teacher_emb)
    #   Epoch   1: ~0.3-0.5  (random init, large vector mismatch)
    #   Epoch  10: ~0.1-0.3  (warmup complete, student converging)
    #   Epoch  50: ~0.01-0.1
    #   Epoch 100: ~0.005-0.05 (plateau)
    #
    # Collapse detection:
    #   student_std should track towards teacher_std (~0.04).
    #   If student_std < teacher_std * 0.2, the student is collapsing.
    #   If student_std > teacher_std * 10, the student is diverging.

    # ----- Metrics logger -----
    metrics_logger = MetricsLogger(args.output_dir)

    print(f"\n{'='*55}")
    print(f"  Starting training from epoch {start_epoch}")
    print(f"{'='*55}\n")

    # Reset peak GPU memory counter for accurate per-run tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        avg_loss, avg_align, avg_var, avg_s_std, avg_t_std = train_one_epoch(
            teacher, student, projector, dataloader, optimizer, args.device,
        )

        # Step the LR scheduler (once per epoch)
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        gpu_mem = get_gpu_memory_mb()

        # ----- Logging -----
        print(
            f"  Epoch [{epoch + 1:3d}/{args.epochs}]  "
            f"loss={avg_loss:.4f}  "
            f"align={avg_align:.4f}  "
            f"var={avg_var:.4f}  "
            f"s_std={avg_s_std:.4f}  "
            f"t_std={avg_t_std:.4f}  "
            f"lr={current_lr:.2e}  "
            f"time={elapsed:.1f}s  "
            f"gpu={gpu_mem['gpu_mem_allocated_mb']:.0f}MB"
        )

        # ----- Log to CSV -----
        metrics_logger.log(
            epoch, avg_loss, avg_s_std, avg_t_std,
            current_lr, elapsed, gpu_mem,
        )

        # ----- Collapse warning (dynamic, relative to teacher) -----
        collapse_floor = avg_t_std * COLLAPSE_RATIO
        if avg_s_std < collapse_floor:
            print(
                f"  [WARNING] Student embedding_std={avg_s_std:.4f} < "
                f"{collapse_floor:.4f} (teacher_std * {COLLAPSE_RATIO}) "
                f"-- possible representation collapse!"
            )
            print(
                f"  [WARNING] Consider stopping training and debugging "
                f"augmentations, learning rate, or loss function."
            )
        elif avg_s_std > avg_t_std * 10:
            print(
                f"  [WARNING] Student embedding_std={avg_s_std:.4f} >> "
                f"teacher_std={avg_t_std:.4f} -- student may be diverging."
            )

        # ----- Save ckpt_latest.pth every epoch (resume point) -----
        save_checkpoint(
            student, projector, optimizer, scheduler,
            epoch, avg_loss, args.output_dir, "ckpt_latest.pth",
        )

        # ----- Save periodic checkpoint -----
        if (epoch + 1) % args.save_every == 0:
            name = f"ckpt_epoch_{epoch + 1:04d}.pth"
            save_checkpoint(
                student, projector, optimizer, scheduler,
                epoch, avg_loss, args.output_dir, name,
            )
            print(f"    -> Saved {name}")

    # ----- Final GPU summary -----
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"\n  Peak GPU memory: {peak:.0f} MB")

    print(f"\n{'='*55}")
    print(f"  Training complete.  Final loss: {avg_loss:.4f}")
    print(f"  Metrics saved to:  {os.path.join(args.output_dir, METRICS_FILENAME)}")
    print(f"  Checkpoints saved: {args.output_dir}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
