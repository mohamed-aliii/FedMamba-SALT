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
  Loss = 1 - cosine_similarity  (range [0, 2])
  Epoch   1:  loss ~ 0.9-1.0  (random init, cosine sim near 0)
  Epoch  10:  loss ~ 0.4-0.7  (warmup complete, student starting to align)
  Epoch  30:  loss ~ 0.15-0.4
  Epoch 100:  loss ~ 0.05-0.2  (plateau)

  embedding_std should stay ABOVE 0.1 throughout training.
  If it drops below 0.05, representations are collapsing --
  stop training and debug (check augmentations, LR, loss).
=========================================
"""

import argparse
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
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from augmentations.medical_aug import DualViewDataset
from models.inception_mamba import InceptionMambaEncoder
from models.vit_teacher import ViTTeacher
from objectives.salt_loss import ProjectionHead, embedding_std, salt_loss

# ======================================================================
# Constants
# ======================================================================
WARMUP_EPOCHS = 10
COLLAPSE_THRESHOLD = 0.05  # embedding_std below this -> likely collapse


# ======================================================================
# CLI
# ======================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FedMamba-SALT: Centralized self-supervised pre-training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument(
        "--data_path", type=str, required=True,
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

    return parser.parse_args()


# ======================================================================
# Data
# ======================================================================
def build_dataloader(args: argparse.Namespace) -> DataLoader:
    """Build the DualViewDataset + DataLoader from an ImageFolder."""
    train_root = os.path.join(args.data_path, "train")
    if not os.path.isdir(train_root):
        print(f"[ERROR] Expected a train/ subdirectory at: {train_root}")
        sys.exit(1)

    base_ds = ImageFolder(root=train_root)
    dual_ds = DualViewDataset(base_ds)

    loader = DataLoader(
        dual_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    print(f"Dataset: {len(base_ds)} images from {train_root}")
    print(f"DataLoader: {len(loader)} batches of {args.batch_size}")
    return loader


# ======================================================================
# Models
# ======================================================================
def build_models(args: argparse.Namespace):
    """Instantiate teacher, student, and projection head."""

    # Teacher: frozen ViT-B/16 (load checkpoint if available)
    ckpt_path = args.teacher_ckpt if os.path.isfile(args.teacher_ckpt) else None
    if ckpt_path is None:
        print("[WARN] Teacher checkpoint not found -- using random weights.")
        print(f"       Expected at: {args.teacher_ckpt}")
    teacher = ViTTeacher(ckpt_path=ckpt_path).to(args.device)

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
    ckpt = torch.load(latest_path, map_location=device)

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
    Returns (avg_loss, avg_student_std, avg_teacher_std).
    """
    student.train()
    projector.train()
    # teacher stays in eval() permanently (enforced inside its forward())

    total_loss = 0.0
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

        # ----- SALT loss -----
        loss = salt_loss(s_proj, t_emb)

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
        batch_loss = loss.item()
        total_loss += batch_loss
        total_s_std += s_std
        total_t_std += t_std
        n_batches += 1

        pbar.set_postfix(loss=f"{batch_loss:.4f}", s_std=f"{s_std:.3f}")

    avg_loss = total_loss / max(1, n_batches)
    avg_s_std = total_s_std / max(1, n_batches)
    avg_t_std = total_t_std / max(1, n_batches)

    return avg_loss, avg_s_std, avg_t_std


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
    # Expected loss trajectory (healthy training):
    #   Loss = 1 - cosine_similarity  (range [0, 2])
    #   Epoch   1: ~0.9-1.0  (random init, cosine sim near 0)
    #   Epoch  10: ~0.4-0.7  (warmup complete, student starting to align)
    #   Epoch  30: ~0.15-0.4
    #   Epoch 100: ~0.05-0.2 (plateau)
    #
    # Collapse detection:
    #   embedding_std measures the average per-dimension standard deviation
    #   of the student projections across a batch.
    #     - Healthy:   > 0.1
    #     - Warning:   0.05 - 0.1
    #     - Collapsed: < 0.05  --> STOP and debug
    #
    #   A collapsed model outputs nearly identical embeddings for all inputs,
    #   meaning it has found a degenerate shortcut instead of learning
    #   meaningful representations.

    print(f"\n{'='*55}")
    print(f"  Starting training from epoch {start_epoch}")
    print(f"{'='*55}\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        avg_loss, avg_s_std, avg_t_std = train_one_epoch(
            teacher, student, projector, dataloader, optimizer, args.device,
        )

        # Step the LR scheduler (once per epoch)
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        # ----- Logging -----
        print(
            f"  Epoch [{epoch + 1:3d}/{args.epochs}]  "
            f"loss={avg_loss:.4f}  "
            f"s_std={avg_s_std:.4f}  "
            f"t_std={avg_t_std:.4f}  "
            f"lr={current_lr:.2e}  "
            f"time={elapsed:.1f}s"
        )

        # ----- Collapse warning -----
        if avg_s_std < COLLAPSE_THRESHOLD:
            print(
                f"  [WARNING] Student embedding_std={avg_s_std:.4f} < {COLLAPSE_THRESHOLD} "
                f"-- possible representation collapse!"
            )
            print(
                f"  [WARNING] Consider stopping training and debugging "
                f"augmentations, learning rate, or loss function."
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

    print(f"\n{'='*55}")
    print(f"  Training complete.  Final loss: {avg_loss:.4f}")
    print(f"  Checkpoints saved to: {args.output_dir}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
