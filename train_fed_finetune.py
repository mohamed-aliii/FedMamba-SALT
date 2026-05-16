#!/usr/bin/env python
"""
train_fed_finetune.py -- Federated fine-tuning evaluation for FedMamba-SALT.

After federated (or centralized) pre-training, this script fine-tunes the
encoder + classifier in a federated fashion: each client trains on its own
labeled split, and the global model is aggregated via FedAvg every round.

Architecture mirrors train_fedavg.py exactly:
  - Same FedAvg / FedProx aggregation logic
  - Same per-client persistent optimizer pattern (SCHEDULER FIX A)
  - Same warmup + flat + cosine LR schedule (SCHEDULER FIX B/C/D)
  - Same early stopping, CSV logging, and checkpoint conventions

Key differences from pre-training:
  - No teacher model; loss is cross-entropy on labeled data
  - Encoder can be frozen (federated_linear_probe) or unfrozen (federated_finetune)
  - Classifier head is aggregated together with the encoder each round
  - Supports AttentionPoolClassifier (patch tokens) or flat Linear head (GAP)
  - Per-round validation on the global test set
  - Per-client label scarcity via --label_fraction

Modes (--mode):
  federated_finetune        Full encoder + classifier fine-tuning, federated
  federated_linear_probe    Encoder frozen; only classifier trains, federated

Usage:
    python train_fed_finetune.py \\
        --encoder_ckpt outputs/fedavg/ckpt_best.pth \\
        --data_path /path/to/dataset \\
        --num_classes 5 \\
        --n_clients 5 \\
        --split_type split_1 \\
        --max_rounds 50 \\
        --mode federated_finetune

Reference:
    - McMahan et al., "Communication-Efficient Learning of Deep Networks
      from Decentralized Data", AISTATS 2017 (FedAvg)
    - Li et al., "Federated Optimization in Heterogeneous Networks"
      (FedProx, ICLR 2020)
"""

import argparse
import copy
import csv
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from augmentations.retina_dataset import RetinaDataset
from augmentations.medical_aug import RETINA_MEAN, RETINA_STD
from models.inception_mamba import InceptionMambaEncoder
from train_centralized import load_yaml_config, get_gpu_memory_mb
from utils.ckpt_compat import safe_torch_load
from utils.fedavg import (
    average_models, broadcast_global_to_clients, compute_client_weights,
)

# Reuse shared components from the linear probe script
from eval.linear_probe import (
    get_eval_transform,
    get_train_transform,
    load_encoder,
    AttentionPoolClassifier,
    PatchEncoderWrapper,
    FocalLoss,
    mixup_data,
    mixup_criterion,
    stratified_subset,
    evaluate_finetune,
    save_confusion_matrix,
    save_roc_curve,
    save_training_curves,
    print_classification_report,
    FINETUNE_PATIENCE,
    MIXUP_ALPHA,
)


# ======================================================================
# Constants
# ======================================================================
METRICS_FILENAME = "fed_finetune_metrics.csv"
LOSS_PATIENCE    = 20      # early stop if global val_acc doesn't improve
ACC_MIN_DELTA    = 0.05    # minimum improvement (%) to reset patience counter


# ======================================================================
# CLI
# ======================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FedMamba-SALT: Federated fine-tuning evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file (same two-pass pattern as train_fedavg.py)
    p.add_argument("--config", type=str, default=None,
                   help="Optional YAML config. CLI flags override it.")

    # Checkpoint
    p.add_argument("--encoder_ckpt", type=str, required=True,
                   help="Pre-trained encoder checkpoint (.pth)")

    # Data
    p.add_argument("--data_path", type=str, default=None,
                   help="Dataset root (SSL-FL format with client CSVs)")
    p.add_argument("--num_classes", type=int, required=True,
                   help="Number of classification classes")

    # Output
    p.add_argument("--output_dir", type=str, default="eval_results/fed_finetune",
                   help="Directory for checkpoints, metrics, and plots")

    # Federated settings
    p.add_argument("--n_clients", type=int, default=5,
                   help="Number of federated clients")
    p.add_argument("--split_type", type=str, default="split_1",
                   help="Data split: split_1 | split_2 | split_3")
    p.add_argument("--max_rounds", type=int, default=50,
                   help="Maximum communication rounds")
    p.add_argument("--E_epoch", type=int, default=2,
                   help="Local fine-tuning epochs per round per client")
    p.add_argument("--mu", type=float, default=0.0,
                   help="FedProx proximal term weight. 0 = FedAvg")

    # Training hyper-parameters
    p.add_argument("--batch_size", type=int, default=64,
                   help="Per-client batch size. 64 is safe on 16 GB GPU")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Base learning rate for the classifier. "
                        "Encoder gets lr/2 (full_finetune mode)")
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=1.0,
                   help="Max gradient norm. 0 = disabled.")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_every", type=int, default=10,
                   help="Save a named checkpoint every N rounds")
    p.add_argument("--device", type=str, default="cuda")

    # Evaluation
    p.add_argument(
        "--mode", type=str, default="federated_finetune",
        choices=["federated_finetune", "federated_linear_probe"],
        help="federated_finetune: encoder + classifier both train; "
             "federated_linear_probe: encoder frozen, only classifier trains",
    )
    p.add_argument(
        "--label_fraction", type=float, default=1.0,
        help="Fraction of each client's labels to use (0.0–1.0). "
             "Stratified sampling preserves class balance.",
    )
    p.add_argument(
        "--use_mixup", action="store_true",
        help="Apply Mixup augmentation during local training",
    )
    p.add_argument(
        "--use_focal_loss", action="store_true",
        help="Use Focal Loss instead of weighted cross-entropy",
    )

    # Two-pass config loading
    known, _ = p.parse_known_args()
    if known.config is not None:
        yaml_dict = load_yaml_config(known.config)
        valid_keys = {a.dest for a in p._actions}
        p.set_defaults(**{k: v for k, v in yaml_dict.items() if k in valid_keys})

    args = p.parse_args()
    if args.data_path is None:
        p.error("--data_path is required (via CLI or YAML config)")
    return args


# ======================================================================
# Per-client DataLoaders (labeled data for fine-tuning)
# ======================================================================
def build_client_dataloaders(args, train_transform) -> tuple:
    """
    Build one labeled DataLoader per client.

    Uses the same SSL-FL path convention as train_fedavg.py:
        data_path / {n_clients}_clients / {split_type} / client_{id}.csv

    Each client's dataset is optionally subsetted via --label_fraction
    using stratified sampling to preserve class balance.
    """
    loaders = []
    dataset_sizes = []

    for client_id in range(1, args.n_clients + 1):
        split_csv = os.path.join(
            f"{args.n_clients}_clients", args.split_type,
            f"client_{client_id}.csv",
        )

        ds = RetinaDataset(
            data_path=args.data_path,
            phase="train",
            split_type="federated",
            split_csv=split_csv,
            transform=train_transform,
        )

        # Stratified label scarcity
        if args.label_fraction < 1.0:
            ds = stratified_subset(ds, args.label_fraction, seed=client_id)

        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            # drop_last prevents BatchNorm1d crash on size-1 tail batches
            drop_last=True,
            persistent_workers=args.num_workers > 0,
        )

        loaders.append(loader)
        dataset_sizes.append(len(ds))
        print(f"  Client {client_id}: {len(ds)} labeled images  "
              f"({len(loader)} batches)")

    return loaders, dataset_sizes


# ======================================================================
# Build global encoder + classifier
# ======================================================================
def build_models(args):
    """
    Load the pre-trained encoder and build a fresh classifier head.

    For federated_finetune:
        encoder  = PatchEncoderWrapper (returns patch tokens)
        classifier = AttentionPoolClassifier
    For federated_linear_probe:
        encoder  = raw InceptionMambaEncoder (returns GAP vector, frozen)
        classifier = BN → Dropout → Linear
    """
    freeze = (args.mode == "federated_linear_probe")
    base_encoder = load_encoder(args.encoder_ckpt, args.device, freeze=freeze)

    if args.mode == "federated_finetune":
        encoder = PatchEncoderWrapper(base_encoder)
        classifier = AttentionPoolClassifier(
            feat_dim=768, num_classes=args.num_classes,
        ).to(args.device)
        nn.init.kaiming_uniform_(classifier.head[2].weight)
        nn.init.zeros_(classifier.head[2].bias)
    else:
        # linear probe: frozen encoder, flat GAP vector
        encoder = base_encoder  # already on device
        classifier = nn.Sequential(
            nn.BatchNorm1d(768),
            nn.Dropout(0.2),
            nn.Linear(768, args.num_classes),
        ).to(args.device)
        nn.init.kaiming_uniform_(classifier[2].weight)
        nn.init.zeros_(classifier[2].bias)

    enc_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    cls_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(f"\n  Encoder trainable:    {enc_params / 1e6:.2f}M params")
    print(f"  Classifier trainable: {cls_params / 1e6:.2f}M params\n")

    return encoder, classifier


# ======================================================================
# FedProx proximal snapshot
# ======================================================================
def snapshot_global_params(encoder, classifier) -> dict:
    """
    Capture a detached copy of global (encoder + classifier) params for
    the FedProx proximal penalty term. Keys are namespaced to avoid collision.
    """
    params = {}
    for name, param in encoder.named_parameters():
        if param.requires_grad:
            params[f"enc.{name}"] = param.detach().clone()
    for name, param in classifier.named_parameters():
        if param.requires_grad:
            params[f"cls.{name}"] = param.detach().clone()
    return params


def fedprox_penalty(encoder, classifier, global_params: dict, mu: float) -> torch.Tensor:
    """
    Compute the FedProx proximal term:
        (mu / 2) * ||w - w_global||^2

    This pulls each client's local parameters back toward the global model,
    reducing client drift in heterogeneous data settings.
    """
    penalty = torch.tensor(0.0, device=next(encoder.parameters()).device)
    for name, param in encoder.named_parameters():
        if param.requires_grad and f"enc.{name}" in global_params:
            g = global_params[f"enc.{name}"].to(param.device)
            penalty = penalty + ((param - g) ** 2).sum()
    for name, param in classifier.named_parameters():
        if param.requires_grad and f"cls.{name}" in global_params:
            g = global_params[f"cls.{name}"].to(param.device)
            penalty = penalty + ((param - g) ** 2).sum()
    return (mu / 2.0) * penalty


# ======================================================================
# Loss function factory
# ======================================================================
def build_criterion(args, device: str) -> nn.Module:
    """
    Build the training loss.

    Default: weighted CrossEntropyLoss (2× weight on non-healthy classes)
    Option:  FocalLoss (--use_focal_loss), useful for severe class imbalance
    """
    weights = [1.0] + [2.0] * (args.num_classes - 1)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    if args.use_focal_loss:
        return FocalLoss(weight=class_weights, gamma=2.0, label_smoothing=0.05)
    return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)


# ======================================================================
# Local fine-tuning for one client, one round
# ======================================================================
def local_train_one_round(
    encoder: nn.Module,
    classifier: nn.Module,
    loader: DataLoader,
    optimizer,
    scaler,
    criterion: nn.Module,
    args,
    global_params: dict | None,
    freeze_encoder: bool,
) -> tuple:
    """
    Run E_epoch local fine-tuning steps for a single client.

    Returns:
        avg_loss (float), train_acc (float)
    """
    _device_type = "cuda" if "cuda" in args.device else "cpu"

    total_loss = 0.0
    correct = 0
    total = 0

    for _local_epoch in range(args.E_epoch):
        if not freeze_encoder:
            encoder.train()
        else:
            encoder.eval()
        classifier.train()

        for images, labels in loader:
            images = images.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)

            # Optional Mixup
            if args.use_mixup:
                images, targets_a, targets_b, lam = mixup_data(
                    images, labels, MIXUP_ALPHA,
                )
            else:
                targets_a, targets_b, lam = labels, labels, 1.0

            optimizer.zero_grad()

            with torch.amp.autocast(_device_type, enabled=("cuda" in args.device)):
                # Always use return_patches=False for the raw encoder path.
                # PatchEncoderWrapper ignores kwargs and calls encoder(x, return_patches=True)
                # internally, so the classifier receives the right tensor shape in both modes.
                if isinstance(encoder, PatchEncoderWrapper):
                    features = encoder(images)
                else:
                    features = encoder(images, return_patches=False)

                logits = classifier(features)

                if args.use_mixup:
                    loss = mixup_criterion(
                        criterion, logits, targets_a, targets_b, lam,
                    )
                else:
                    loss = criterion(logits, labels)

                # FedProx proximal term
                if global_params is not None and args.mu > 0:
                    loss = loss + fedprox_penalty(
                        encoder, classifier, global_params, args.mu,
                    )

            scaler.scale(loss).backward()

            # Gradient clipping on active params only (skip frozen encoder)
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                active = [
                    p for p in list(encoder.parameters()) + list(classifier.parameters())
                    if p.requires_grad and p.grad is not None
                ]
                torch.nn.utils.clip_grad_norm_(active, max_norm=args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * images.size(0)
            dom_labels = targets_a if lam >= 0.5 else targets_b
            correct += (logits.argmax(dim=1) == dom_labels).sum().item()
            total += images.size(0)

    avg_loss = total_loss / max(total, 1)
    train_acc = 100.0 * correct / max(total, 1)
    return avg_loss, train_acc


# ======================================================================
# Global evaluation (test set, no TTA for speed)
# ======================================================================
@torch.no_grad()
def evaluate_global(
    encoder: nn.Module,
    classifier: nn.Module,
    loader: DataLoader,
    device: str,
    num_classes: int,
) -> tuple:
    """
    Fast global evaluation used during the federated training loop.
    No TTA (speed matters across many rounds).

    Returns:
        (val_acc %, val_loss, auc)
    """
    from sklearn.metrics import roc_auc_score

    encoder.eval()
    classifier.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")

    correct = 0
    total = 0
    total_loss = 0.0
    all_probs = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if isinstance(encoder, PatchEncoderWrapper):
            features = encoder(images)
        else:
            features = encoder(images, return_patches=False)

        logits = classifier(features)
        total_loss += criterion(logits, labels).item()
        probs = torch.softmax(logits, dim=1)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())

    val_acc  = 100.0 * correct / max(total, 1)
    val_loss = total_loss / max(total, 1)

    all_probs_np  = torch.cat(all_probs).numpy()
    all_labels_np = torch.cat(all_labels).numpy()

    try:
        if num_classes == 2:
            auc = roc_auc_score(all_labels_np, all_probs_np[:, 1])
        else:
            auc = roc_auc_score(
                all_labels_np, all_probs_np,
                multi_class="ovr", average="macro",
            )
    except ValueError:
        auc = 0.0

    return val_acc, val_loss, auc


# ======================================================================
# Checkpoint helpers
# ======================================================================
def save_checkpoint(
    encoder, classifier,
    comm_round, val_acc, output_dir, name,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    torch.save({
        "comm_round": comm_round,
        "val_acc":    val_acc,
        "encoder_state_dict":    encoder.state_dict(),
        "classifier_state_dict": classifier.state_dict(),
    }, path)


def try_resume(output_dir, encoder, classifier, device) -> int:
    """Resume from ckpt_latest.pth if present. Returns the starting round."""
    latest = os.path.join(output_dir, "ckpt_latest.pth")
    if not os.path.isfile(latest):
        return 0
    print(f"[RESUME] Loading {latest}")
    ckpt = safe_torch_load(latest, map_location=device)
    if "encoder_state_dict" not in ckpt:
        return 0
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    classifier.load_state_dict(ckpt["classifier_state_dict"])
    start = ckpt["comm_round"] + 1
    print(f"[RESUME] Resuming from round {start}")
    return start


# ======================================================================
# Metrics logger
# ======================================================================
class FedFinetuneLogger:
    BASE_COLS = [
        "round", "val_acc", "val_loss", "auc",
        "enc_lr", "cls_lr", "round_time_s", "gpu_mb",
    ]

    def __init__(self, output_dir, n_clients):
        self.path = os.path.join(output_dir, METRICS_FILENAME)
        cols = self.BASE_COLS + [f"client_{i}_loss" for i in range(1, n_clients + 1)]
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(cols)

    def log(self, comm_round, val_acc, val_loss, auc,
            enc_lr, cls_lr, round_time, gpu_mb, client_losses):
        with open(self.path, "a", newline="") as f:
            row = [
                comm_round + 1,
                f"{val_acc:.2f}", f"{val_loss:.6f}", f"{auc:.4f}",
                f"{enc_lr:.2e}", f"{cls_lr:.2e}",
                f"{round_time:.1f}", f"{gpu_mb:.0f}",
            ] + [f"{cl:.6f}" for cl in client_losses]
            csv.writer(f).writerow(row)


# ======================================================================
# LR schedule (mirrors train_fedavg.py exactly)
# ======================================================================
def compute_round_lr(comm_round: int, max_rounds: int, base_lr: float,
                     mu: float) -> float:
    """
    Three-phase LR schedule:
      Phase 1 — Warmup (0 → WARMUP_ROUNDS): lr/5 → lr  (linear)
      Phase 2 — Flat   (WARMUP → FLAT):      lr          (constant)
      Phase 3 — Cosine (FLAT → max_rounds):  lr → eta_min

    FedProx uses a shorter flat phase (15% vs 25%) to spend more budget
    in the cosine phase, compensating for the extra regularization from mu.
    """
    WARMUP_ROUNDS = 5
    FLAT_RATIO    = 0.15 if mu > 0 else 0.25
    FLAT_ROUNDS   = WARMUP_ROUNDS + int(max_rounds * FLAT_RATIO)
    eta_min       = base_lr * 0.02

    if comm_round < WARMUP_ROUNDS:
        return base_lr * (comm_round + 1) / WARMUP_ROUNDS
    elif comm_round < FLAT_ROUNDS:
        return base_lr
    else:
        t_cur   = comm_round - FLAT_ROUNDS
        T_decay = max(1, max_rounds - FLAT_ROUNDS)
        return eta_min + 0.5 * (base_lr - eta_min) * (
            1 + math.cos(math.pi * t_cur / T_decay)
        )


# ======================================================================
# Main federated fine-tuning loop
# ======================================================================
def main() -> None:
    args = parse_args()
    algo_name = "FedProx" if args.mu > 0 else "FedAvg"
    mode_label = (
        "Federated Full Fine-tune" if args.mode == "federated_finetune"
        else "Federated Linear Probe"
    )
    freeze_encoder = (args.mode == "federated_linear_probe")

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"  FedMamba-SALT: {mode_label} ({algo_name})")
    print("=" * 60)
    print(f"  Encoder ckpt:   {args.encoder_ckpt}")
    print(f"  Split:          {args.split_type}")
    print(f"  Clients:        {args.n_clients}")
    print(f"  Rounds:         {args.max_rounds}")
    print(f"  E_epoch:        {args.E_epoch}")
    print(f"  mu (FedProx):   {args.mu}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  LR:             {args.lr}")
    print(f"  Label fraction: {args.label_fraction:.0%}")
    print(f"  Mixup:          {args.use_mixup}")
    print(f"  Focal Loss:     {args.use_focal_loss}")
    print(f"  Output:         {args.output_dir}")
    print()

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    print("[1/4] Building data loaders...")
    train_transform = (
        get_train_transform(dataset="retina") if not freeze_encoder
        else get_eval_transform(dataset="retina")   # no augmentation for frozen encoder
    )
    eval_transform = get_eval_transform(dataset="retina")

    client_loaders, dataset_sizes = build_client_dataloaders(args, train_transform)
    client_weights = compute_client_weights(dataset_sizes)
    print(f"  Client weights: {[f'{w:.3f}' for w in client_weights]}")

    test_ds = RetinaDataset(
        data_path=args.data_path,
        phase="test",
        split_type="central",
        split_csv="test.csv",
        transform=eval_transform,
    )
    class_names = test_ds.classes
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f"  Test: {len(test_ds)} images  |  Classes: {class_names}\n")

    # ------------------------------------------------------------------
    # 2. Models
    # ------------------------------------------------------------------
    print("[2/4] Building global encoder + classifier...")
    global_encoder, global_classifier = build_models(args)

    # Resume if a checkpoint exists
    start_round = try_resume(
        args.output_dir, global_encoder, global_classifier, args.device,
    )

    # ------------------------------------------------------------------
    # 3. Per-client copies
    # ------------------------------------------------------------------
    print("[3/4] Creating per-client model copies...")
    client_encoders    = [copy.deepcopy(global_encoder)    for _ in range(args.n_clients)]
    client_classifiers = [copy.deepcopy(global_classifier) for _ in range(args.n_clients)]

    broadcast_global_to_clients(global_encoder,    client_encoders)
    broadcast_global_to_clients(global_classifier, client_classifiers)

    # Persistent per-client optimizers (SCHEDULER FIX A from train_fedavg.py):
    # Re-creating the optimizer every round discards AdamW momentum buffers,
    # making late-round cosine LR reductions ineffective.
    def _make_optimizer(enc, cls):
        enc_params = [p for p in enc.parameters() if p.requires_grad]
        cls_params = [p for p in cls.parameters() if p.requires_grad]
        # Separate LR groups: encoder gets lr/2 to protect pre-trained features.
        # Classifier gets full lr (random init needs faster movement).
        # Linear probe: encoder has requires_grad=False so enc_params is empty.
        param_groups = []
        if enc_params:
            param_groups.append({
                "params": enc_params,
                "lr": args.lr / 2.0,
                "weight_decay": 0.01,
            })
        param_groups.append({
            "params": cls_params,
            "lr": args.lr,
            "weight_decay": 0.05,
        })
        return AdamW(param_groups, betas=(0.9, 0.999))

    client_optimizers = [
        _make_optimizer(client_encoders[i], client_classifiers[i])
        for i in range(args.n_clients)
    ]
    client_scalers = [
        torch.amp.GradScaler(
            "cuda" if "cuda" in args.device else "cpu",
            enabled=("cuda" in args.device),
        )
        for _ in range(args.n_clients)
    ]

    criterion = build_criterion(args, args.device)
    logger    = FedFinetuneLogger(args.output_dir, args.n_clients)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # ------------------------------------------------------------------
    # 4. Federated fine-tuning loop
    # ------------------------------------------------------------------
    print(f"[4/4] Starting federated fine-tuning from round {start_round}...")
    _warmup_r = 5
    _flat_r   = _warmup_r + int(args.max_rounds * (0.15 if args.mu > 0 else 0.25))
    print(f"  LR schedule:   warmup={_warmup_r}r | flat={_flat_r - _warmup_r}r "
          f"| cosine={args.max_rounds - _flat_r}r | eta_min={args.lr * 0.02:.1e}")
    print(f"  Early stopping: no improvement for {LOSS_PATIENCE} rounds "
          f"(threshold >{ACC_MIN_DELTA:.2f}%)")
    print("=" * 60)

    best_acc          = 0.0
    best_round        = 0
    rounds_no_improve = 0
    history = {
        "round": [], "val_acc": [], "val_loss": [],
        "val_auc": [], "enc_lr": [], "cls_lr": [],
    }

    for comm_round in range(start_round, args.max_rounds):
        round_start = time.time()

        # ---- Compute current LR ----
        current_lr = compute_round_lr(comm_round, args.max_rounds, args.lr, args.mu)
        # Encoder gets lr/2; classifier gets full lr (mirrors _make_optimizer)
        enc_lr = current_lr / 2.0
        cls_lr = current_lr

        # ---- FedProx: snapshot global params before any client trains ----
        global_params = None
        if args.mu > 0:
            global_params = snapshot_global_params(global_encoder, global_classifier)

        client_losses    = []
        client_train_acc = []

        # ---- Local training for each client ----
        for cid in range(args.n_clients):
            enc   = client_encoders[cid]
            cls   = client_classifiers[cid]
            opt   = client_optimizers[cid]
            scl   = client_scalers[cid]

            # Update LR in-place (preserves momentum buffers, see SCHEDULER FIX A)
            for pg in opt.param_groups:
                # param_groups[0] = encoder (if unfrozen), param_groups[-1] = classifier
                if "weight_decay" in pg and pg["weight_decay"] == 0.01:
                    pg["lr"] = enc_lr
                else:
                    pg["lr"] = cls_lr

            loss, tacc = local_train_one_round(
                enc, cls, client_loaders[cid], opt, scl,
                criterion, args, global_params, freeze_encoder,
            )
            client_losses.append(loss)
            client_train_acc.append(tacc)

        # ---- FedAvg aggregation ----
        # Aggregate encoder and classifier independently using dataset-size weights.
        # This is safe even when the encoder is frozen: averaging frozen params
        # is a no-op (all clients share identical weights), and the classifier
        # aggregation is what matters in linear probe mode.
        average_models(global_encoder,    client_encoders,    client_weights)
        average_models(global_classifier, client_classifiers, client_weights)

        # ---- Broadcast back ----
        broadcast_global_to_clients(global_encoder,    client_encoders)
        broadcast_global_to_clients(global_classifier, client_classifiers)

        # ---- Global evaluation ----
        val_acc, val_loss, val_auc = evaluate_global(
            global_encoder, global_classifier,
            test_loader, args.device, args.num_classes,
        )

        round_time = time.time() - round_start
        gpu_mb     = get_gpu_memory_mb()["gpu_mem_allocated_mb"]

        # ---- Checkpoint on improvement ----
        is_best = val_acc > best_acc + ACC_MIN_DELTA
        if is_best:
            best_acc   = val_acc
            best_round = comm_round + 1
            rounds_no_improve = 0
            save_checkpoint(
                global_encoder, global_classifier,
                comm_round, val_acc,
                args.output_dir, "ckpt_best_finetune.pth",
            )
        else:
            rounds_no_improve += 1

        save_checkpoint(
            global_encoder, global_classifier,
            comm_round, val_acc,
            args.output_dir, "ckpt_latest.pth",
        )
        if (comm_round + 1) % args.save_every == 0:
            name = f"ckpt_round_{comm_round + 1:04d}.pth"
            save_checkpoint(
                global_encoder, global_classifier,
                comm_round, val_acc, args.output_dir, name,
            )

        # ---- History & logging ----
        history["round"].append(comm_round + 1)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["enc_lr"].append(enc_lr)
        history["cls_lr"].append(cls_lr)

        logger.log(
            comm_round, val_acc, val_loss, val_auc,
            enc_lr, cls_lr, round_time, gpu_mb, client_losses,
        )

        # ---- Console output ----
        client_loss_str = "  ".join(
            f"c{i+1}={client_losses[i]:.4f}" for i in range(args.n_clients)
        )
        marker = " *BEST*" if is_best else ""
        print(
            f"  Round [{comm_round + 1:3d}/{args.max_rounds}]  "
            f"val={val_acc:.2f}%  auc={val_auc:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"enc_lr={enc_lr:.1e}  cls_lr={cls_lr:.1e}  "
            f"time={round_time:.1f}s  {client_loss_str}{marker}"
        )

        # ---- Early stopping ----
        if rounds_no_improve >= LOSS_PATIENCE:
            print(
                f"\n  [EARLY STOP] No improvement > {ACC_MIN_DELTA:.2f}% for "
                f"{LOSS_PATIENCE} rounds. Best: {best_acc:.2f}% at round {best_round}."
            )
            break

        if math.isnan(val_loss):
            print(f"\n  [ABORT] Round {comm_round + 1}: val_loss is NaN. Stopping.")
            break

    # ------------------------------------------------------------------
    # Final evaluation with TTA on best checkpoint
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  Loading best checkpoint for final evaluation (with TTA)...")
    best_path = os.path.join(args.output_dir, "ckpt_best_finetune.pth")
    if os.path.exists(best_path):
        ckpt = safe_torch_load(best_path, map_location=args.device)
        global_encoder.load_state_dict(ckpt["encoder_state_dict"])
        global_classifier.load_state_dict(ckpt["classifier_state_dict"])
        print(f"  Loaded best checkpoint (round {best_round}, "
              f"val_acc={best_acc:.2f}%)")

    top1, per_class, cm, all_probs, all_labels = evaluate_finetune(
        global_encoder, global_classifier,
        test_loader, args.num_classes, args.device, class_names,
    )

    # AUC on TTA predictions
    from sklearn.metrics import roc_auc_score
    try:
        if args.num_classes == 2:
            final_auc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            final_auc = roc_auc_score(
                all_labels, all_probs, multi_class="ovr", average="macro",
            )
    except ValueError:
        final_auc = 0.0

    # ------------------------------------------------------------------
    # Save plots and reports
    # ------------------------------------------------------------------
    mode_tag = f"{args.mode}_{args.split_type}"

    cm_path = os.path.join(args.output_dir, f"confusion_matrix_{mode_tag}.png")
    save_confusion_matrix(cm, class_names, cm_path,
                          title=f"{mode_label} Confusion Matrix")

    save_roc_curve(all_probs, all_labels, class_names, args.output_dir, mode_tag)
    print_classification_report(cm, class_names, args.output_dir, mode_tag)

    # Training curve (val_acc / val_loss / AUC across rounds)
    _save_round_curves(history, args.output_dir, mode_tag)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"\n  Peak GPU memory: {peak_mb:.0f} MB")

    print(f"\n{'='*60}")
    print(f"  {mode_label} Results ({algo_name})")
    print(f"{'='*60}")
    print(f"  Top-1 Accuracy (TTA): {top1:.2f}%")
    print(f"  AUC Score (TTA):      {final_auc:.4f}")
    print(f"  Best round:           {best_round} / {args.max_rounds}")
    print(f"  Clients:              {args.n_clients} ({args.split_type})")
    if args.label_fraction < 1.0:
        print(f"  Label fraction:       {args.label_fraction:.0%}")
    print("\n  Per-class accuracy:")
    for name, acc in per_class.items():
        print(f"    {name:>20s}: {acc:.2f}%")
    print(f"{'='*60}\n")


# ======================================================================
# Round-level training curves
# ======================================================================
def _save_round_curves(history: dict, output_dir: str, mode_tag: str) -> None:
    """Save val_acc, val_loss, AUC, and LR curves across rounds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rounds = history["round"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Federated Fine-Tune: Per-Round Metrics", fontsize=14, fontweight="bold")

    # Val accuracy
    ax = axes[0]
    ax.plot(rounds, history["val_acc"], color="#2ecc71", linewidth=2)
    if history["val_acc"]:
        best_idx = int(np.argmax(history["val_acc"]))
        ax.axvline(x=rounds[best_idx], color="#27ae60", linestyle="--", alpha=0.6)
        ax.annotate(
            f"Best: {history['val_acc'][best_idx]:.1f}%",
            xy=(rounds[best_idx], history["val_acc"][best_idx]),
            fontsize=9, fontweight="bold", color="#27ae60",
        )
    ax.set_xlabel("Round")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Global Validation Accuracy")
    ax.grid(True, alpha=0.3)

    # AUC
    ax = axes[1]
    ax.plot(rounds, history["val_auc"], color="#8e44ad", linewidth=2)
    ax.set_xlabel("Round")
    ax.set_ylabel("AUC")
    ax.set_title("Global Validation AUC")
    ax.grid(True, alpha=0.3)

    # LR
    ax = axes[2]
    ax.plot(rounds, history["enc_lr"], label="Encoder LR", color="#9b59b6", linewidth=2)
    ax.plot(rounds, history["cls_lr"], label="Classifier LR", color="#f39c12", linewidth=2)
    ax.set_xlabel("Round")
    ax.set_ylabel("Learning Rate")
    ax.set_title("LR Schedule")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(output_dir, f"round_curves_{mode_tag}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Round curves saved to: {path}")


if __name__ == "__main__":
    main()
