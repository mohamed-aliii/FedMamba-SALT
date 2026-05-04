#!/usr/bin/env python
"""
train_fedavg.py -- Federated SALT pre-training for FedMamba-SALT.

Orchestrates FedAvg / FedProx across N clients using sequential simulation.
Each communication round:
    1. For each client: load client data, train locally for E epochs
    2. Weighted-average all client models into a global model
    3. Broadcast global model back to all clients

Supports both FedAvg (mu=0) and FedProx (mu>0) via the --mu flag.

Usage:
    # FedAvg on split_1
    python train_fedavg.py --config configs/retina_fedavg.yaml --split_type split_1

    # FedProx on split_1
    python train_fedavg.py --config configs/retina_fedavg.yaml --split_type split_1 --mu 0.01

Reference:
    - SSL-FL (Yan et al.) run_mae_pretrain_FedAvg.py
"""

import argparse
import copy
import csv
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR   # ← ADD THIS
from torch.utils.data import DataLoader

from augmentations.medical_aug import (
    DualViewDataset, get_teacher_transform, get_student_transform,
)
from augmentations.retina_dataset import RetinaDataset
from models.inception_mamba import InceptionMambaEncoder
from models.vit_teacher import FrozenViTTeacher
from objectives.salt_loss import ProjectionHead, embedding_std
from train_centralized import (
    train_one_epoch, load_yaml_config, get_gpu_memory_mb,
)
from utils.ckpt_compat import safe_torch_load
from utils.fedavg import (
    average_models, broadcast_global_to_clients, compute_client_weights,
)


# ======================================================================
# Constants
# ======================================================================
METRICS_FILENAME = "federated_metrics.csv"
LOSS_PATIENCE = 25       # stop if loss doesn't improve for this many rounds
LOSS_MIN_DELTA = 1e-5    # minimum improvement to count as progress


# ======================================================================
# CLI
# ======================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FedMamba-SALT: Federated SALT pre-training (FedAvg / FedProx)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file
    parser.add_argument("--config", type=str, default=None)

    # Data
    parser.add_argument("--data_path", type=str, default=None,
                        help="Root of the dataset (e.g. data/Retina)")
    parser.add_argument("--teacher_ckpt", type=str,
                        default="data/ckpts/mae_vit_base.pth")

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/fedavg")

    # Federated settings
    parser.add_argument("--n_clients", type=int, default=5,
                        help="Number of clients per split")
    parser.add_argument("--split_type", type=str, default="split_1",
                        help="Data split: split_1, split_2, split_3")
    parser.add_argument("--max_rounds", type=int, default=200,
                        help="Total communication rounds")
    parser.add_argument("--E_epoch", type=int, default=1,
                        help="Local training epochs per round per client")
    parser.add_argument("--mu", type=float, default=0.0,
                        help="FedProx proximal penalty. 0 = FedAvg, >0 = FedProx")

    # Training hyper-parameters (per-client)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mask_ratio", type=float, default=0.5,
                        help="Internal latent masking ratio for student")

    # Two-pass config loading (same pattern as train_centralized.py)
    known, _ = parser.parse_known_args()
    if known.config is not None:
        yaml_dict = load_yaml_config(known.config)
        valid_keys = {a.dest for a in parser._actions}
        filtered = {k: v for k, v in yaml_dict.items() if k in valid_keys}
        parser.set_defaults(**filtered)

    args = parser.parse_args()
    if args.data_path is None:
        parser.error("--data_path is required (via CLI or YAML config)")
    return args


# ======================================================================
# Build per-client DataLoaders
# ======================================================================
def build_client_dataloaders(args) -> list:
    """
    Build one DataLoader per client using the split CSVs.

    Path convention (matches SSL-FL):
        data_path/5_clients/split_1/client_1.csv
        data_path/5_clients/split_1/client_2.csv
        ...
    """
    loaders = []
    dataset_sizes = []

    for client_id in range(1, args.n_clients + 1):
        # Construct the CSV path relative to data_path
        split_csv = os.path.join(
            f"{args.n_clients}_clients", args.split_type,
            f"client_{client_id}.csv",
        )

        base_ds = RetinaDataset(
            data_path=args.data_path,
            phase="train",
            split_type="federated",  # anything non-"central" triggers raw path
            split_csv=split_csv,
        )

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

        loaders.append(loader)
        dataset_sizes.append(len(base_ds))
        print(f"  Client {client_id}: {len(base_ds)} images, "
              f"{len(loader)} batches")

    return loaders, dataset_sizes


# ======================================================================
# Build models (reuses the same architecture as centralized)
# ======================================================================
def build_models(args):
    """Instantiate teacher, global student, global projector."""
    if not os.path.isfile(args.teacher_ckpt):
        raise FileNotFoundError(
            f"Teacher checkpoint not found: {args.teacher_ckpt}"
        )
    teacher = FrozenViTTeacher(ckpt_path=args.teacher_ckpt).to(args.device)

    student = InceptionMambaEncoder(
        patch_size=16, embed_dim=384, depth=6, out_dim=768,
    ).to(args.device)

    projector = ProjectionHead(
        in_dim=768, hidden_dim=2048, out_dim=768,
    ).to(args.device)

    student_params = sum(p.numel() for p in student.parameters())
    proj_params = sum(p.numel() for p in projector.parameters())
    print(f"\n{'='*55}")
    print(f"  Teacher (frozen): "
          f"{sum(p.numel() for p in teacher.parameters()) / 1e6:.2f}M params")
    print(f"  Student encoder:  {student_params / 1e6:.2f}M trainable params")
    print(f"  Projection head:  {proj_params / 1e6:.2f}M trainable params")
    print(f"{'='*55}\n")

    return teacher, student, projector


# ======================================================================
# Snapshot global params for FedProx
# ======================================================================
def snapshot_global_params(student, projector):
    """
    Create a detached copy of global model params for FedProx proximal term.
    Keys are prefixed so student and projector params don't collide.
    """
    params = {}
    for name, param in student.named_parameters():
        if param.requires_grad:
            params[name] = param.detach().clone()
    for name, param in projector.named_parameters():
        if param.requires_grad:
            params[f"proj.{name}"] = param.detach().clone()
    return params


# ======================================================================
# Checkpointing
# ======================================================================
def save_fed_checkpoint(
    global_student, global_projector,
    comm_round, loss, output_dir, name,
):
    """Save a federated training checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    torch.save({
        "comm_round": comm_round,
        "loss": loss,
        "dense_distillation": True,
        "student_state_dict": global_student.state_dict(),
        "projector_state_dict": global_projector.state_dict(),
    }, path)


def try_resume_fed(output_dir, global_student, global_projector, device):
    """Resume from ckpt_latest.pth if it exists. Returns start_round."""
    latest = os.path.join(output_dir, "ckpt_latest.pth")
    if not os.path.isfile(latest):
        return 0

    print(f"[RESUME] Loading: {latest}")
    ckpt = safe_torch_load(latest, map_location=device)
    if "student_state_dict" not in ckpt:
        return 0

    global_student.load_state_dict(ckpt["student_state_dict"])
    global_projector.load_state_dict(ckpt["projector_state_dict"])

    start_round = ckpt["comm_round"] + 1
    print(f"[RESUME] Resuming from round {start_round}")
    return start_round


# ======================================================================
# Federated Metrics Logger
# ======================================================================
class FedMetricsLogger:
    COLUMNS = [
        "round", "avg_loss", "avg_enc_std", "lr",
        "round_time_s", "gpu_mb",
    ] + [f"client_{i}_loss" for i in range(1, 6)]

    def __init__(self, output_dir, n_clients):
        self.path = os.path.join(output_dir, METRICS_FILENAME)
        self.n_clients = n_clients
        cols = self.COLUMNS[:6] + [
            f"client_{i}_loss" for i in range(1, n_clients + 1)
        ]
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(cols)

    def log(self, round_num, avg_loss, avg_enc_std, lr,
            round_time, gpu_mb, client_losses):
        with open(self.path, "a", newline="") as f:
            row = [
                round_num + 1, f"{avg_loss:.6f}", f"{avg_enc_std:.4f}",
                f"{lr:.2e}", f"{round_time:.1f}", f"{gpu_mb:.0f}",
            ] + [f"{cl:.6f}" for cl in client_losses]
            csv.writer(f).writerow(row)


# ======================================================================
# Main
# ======================================================================
def main():
    args = parse_args()
    algo_name = "FedProx" if args.mu > 0 else "FedAvg"

    print("=" * 55)
    print(f"  FedMamba-SALT: Federated Pre-training ({algo_name})")
    print("=" * 55)
    print(f"  Split:      {args.split_type}")
    print(f"  Clients:    {args.n_clients}")
    print(f"  Rounds:     {args.max_rounds}")
    print(f"  E_epoch:    {args.E_epoch}")
    print(f"  mu:         {args.mu}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR:         {args.lr}")
    print(f"  Device:     {args.device}")
    print(f"  Output:     {args.output_dir}")
    print()

    os.makedirs(args.output_dir, exist_ok=True)

    # ----- Build components -----
    print("[1/4] Building client dataloaders...")
    client_loaders, dataset_sizes = build_client_dataloaders(args)
    client_weights = compute_client_weights(dataset_sizes)
    print(f"  Client weights: {[f'{w:.3f}' for w in client_weights]}")
    print()

    print("[2/4] Building models...")
    teacher, global_student, global_projector = build_models(args)

    # ----- Resume -----
    start_round = try_resume_fed(
        args.output_dir, global_student, global_projector, args.device,
    )

    # ----- Create per-client model copies -----
    print("[3/4] Creating client model copies...")
    client_students = [
        copy.deepcopy(global_student) for _ in range(args.n_clients)
    ]
    client_projectors = [
        copy.deepcopy(global_projector) for _ in range(args.n_clients)
    ]

    # Broadcast global params to all clients
    broadcast_global_to_clients(global_student, client_students)
    broadcast_global_to_clients(global_projector, client_projectors)

    # ----- Metrics logger -----
    logger = FedMetricsLogger(args.output_dir, args.n_clients)

    # ----- AMP Scaler -----
    # REMOVE this from line 343:
    # scaler = torch.amp.GradScaler("cuda", enabled=(args.device == "cuda"))
    

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print(f"\n[4/4] Starting federated training from round {start_round}...")
    print(f"  Algorithm: {algo_name}")
    print(f"  Early stopping: loss patience={LOSS_PATIENCE}")
    print("=" * 55)
    print()

    # ----- Early stopping state -----
    best_loss = float("inf")
    rounds_no_improve = 0

    # ================================================================
    # Federated Training Loop
    # ================================================================
    for comm_round in range(start_round, args.max_rounds):
        round_start = time.time()
        client_losses = []
        client_enc_stds = []

        # Snapshot global params for FedProx (before any client trains)
        if args.mu > 0:
            global_params = snapshot_global_params(
                global_student, global_projector,
            )
        else:
            global_params = None

        # ----- Local training for each client -----
        for client_id in range(args.n_clients):
            client_student = client_students[client_id]
            client_projector = client_projectors[client_id]
            client_loader = client_loaders[client_id]

            # Per-client optimizer (fresh each round, matching SSL-FL)
            client_params = (
                list(client_student.parameters())
                + list(client_projector.parameters())
            )
            optimizer = AdamW(
                client_params, lr=args.lr, weight_decay=args.weight_decay,
            )

            # Cosine decay: lr → eta_min over remaining rounds
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=args.max_rounds - start_round,
                eta_min=5e-5,
                last_epoch=comm_round - start_round - 1,
            )
            
            #Fresh scaler per client — fixes shared-state corruption
            scaler = torch.amp.GradScaler("cuda", enabled=(args.device == "cuda"))

            # Local E epochs
            for local_epoch in range(args.E_epoch):
                metrics = train_one_epoch(
                    teacher, client_student, client_projector,
                    client_loader, optimizer, scaler, args.device,
                    global_params=global_params,
                    mu=args.mu,
                    mask_ratio=args.mask_ratio,
                )
                avg_loss = metrics[0]
                avg_enc_std = metrics[3]

            client_losses.append(avg_loss)
            client_enc_stds.append(avg_enc_std)

        # ----- FedAvg aggregation -----
        average_models(global_student, client_students, client_weights)
        average_models(global_projector, client_projectors, client_weights)

        # ----- Broadcast back to clients -----
        broadcast_global_to_clients(global_student, client_students)
        broadcast_global_to_clients(global_projector, client_projectors)

        scheduler.step()

        # ----- Round metrics -----
        round_loss = sum(
            w * l for w, l in zip(client_weights, client_losses)
        )
        round_enc_std = sum(
            w * s for w, s in zip(client_weights, client_enc_stds)
        )
        round_time = time.time() - round_start
        gpu = get_gpu_memory_mb()

        # NaN check
        if math.isnan(round_loss):
            print(f"\n  [ABORT] Round {comm_round + 1}: Loss is NaN. Stopping.")
            break

        # Loss plateau early stopping
        if round_loss < best_loss - LOSS_MIN_DELTA:
            best_loss = round_loss
            rounds_no_improve = 0
        else:
            rounds_no_improve += 1
            if rounds_no_improve >= LOSS_PATIENCE:
                print(
                    f"\n  [EARLY STOP] Loss has not improved for {LOSS_PATIENCE} "
                    f"rounds (best={best_loss:.6f}). Stopping training."
                )
                break

        # Collapse check
        if round_enc_std < 0.02:
            print(
                f"  [WARNING] Round {comm_round + 1}: "
                f"enc_std={round_enc_std:.4f} < 0.02 — possible collapse!"
            )

        # ----- Logging -----
        client_loss_str = "  ".join(
            f"c{i+1}={client_losses[i]:.4f}"
            for i in range(args.n_clients)
        )
        print(
            f"  Round [{comm_round + 1:3d}/{args.max_rounds}]  "
            f"loss={round_loss:.4f}  "
            f"enc_std={round_enc_std:.4f}  "
            f"time={round_time:.1f}s  "
            f"{client_loss_str}"
        )

        logger.log(
            comm_round, round_loss, round_enc_std,
            scheduler.get_last_lr()[0], # logs actual decayed LR, not static args.lr
            round_time, gpu["gpu_mem_allocated_mb"], client_losses,
        )

        # ----- Save checkpoint -----
        save_fed_checkpoint(
            global_student, global_projector,
            comm_round, round_loss, args.output_dir, "ckpt_latest.pth",
        )

        if (comm_round + 1) % args.save_every == 0:
            name = f"ckpt_round_{comm_round + 1:04d}.pth"
            save_fed_checkpoint(
                global_student, global_projector,
                comm_round, round_loss, args.output_dir, name,
            )
            print(f"    -> Saved {name}")

    # ----- Summary -----
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"\n  Peak GPU memory: {peak:.0f} MB")

    print(f"\n{'='*55}")
    print(f"  Federated training complete ({algo_name})")
    print(f"  Split:       {args.split_type}")
    print(f"  Rounds:      {args.max_rounds}")
    print(f"  Checkpoints: {args.output_dir}")
    print(f"  Metrics CSV: {os.path.join(args.output_dir, METRICS_FILENAME)}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
