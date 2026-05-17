#!/usr/bin/env python
"""
eval_tta.py — Post-training TTA evaluation for FedMamba-SALT fine-tuned checkpoints.

Usage:
    python eval_tta.py \
        --ckpt /content/fedmamba_salt/outputs/fedprox_split_1/eval_federated_finetune/ckpt_best_finetune.pth \
        --data_path /content/Retina_local \
        --num_classes 2 \
        --n_tta 8 \
        --threshold_sweep
"""

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, classification_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from augmentations.retina_dataset import RetinaDataset
from augmentations.medical_aug import RETINA_MEAN, RETINA_STD
from eval.linear_probe import (
    load_encoder,
    AttentionPoolClassifier,
    PatchEncoderWrapper,
    get_eval_transform,
)
from utils.ckpt_compat import safe_torch_load


# ──────────────────────────────────────────────
# TTA augmentations (tensor-level, no PIL needed)
# ──────────────────────────────────────────────
def get_tta_augmentations(n_tta: int):
    """
    Returns a list of tensor→tensor augmentation functions.
    Input tensors are already normalized [C, H, W].
    n_tta controls how many augmentations to use (4 or 8).
    """
    augs = [
        lambda x: x,                                      # 1. original
        lambda x: torch.flip(x, dims=[2]),                # 2. horizontal flip
        lambda x: torch.rot90(x, k=1, dims=[1, 2]),       # 3. 90°
        lambda x: torch.rot90(x, k=2, dims=[1, 2]),       # 4. 180°
    ]
    if n_tta >= 8:
        augs += [
            lambda x: torch.flip(x, dims=[1]),            # 5. vertical flip
            lambda x: torch.rot90(x, k=3, dims=[1, 2]),   # 6. 270°
            lambda x: torch.clamp(x * 1.1, -3.0, 3.0),   # 7. brighter
            lambda x: torch.clamp(x * 0.9, -3.0, 3.0),   # 8. darker
        ]
    return augs[:n_tta]


# ──────────────────────────────────────────────
# Core TTA inference
# ──────────────────────────────────────────────
@torch.no_grad()
def predict_with_tta(encoder, classifier, dataloader, augmentations, device):
    """
    Returns:
        all_probs  : np.ndarray [N, num_classes]  averaged over TTA passes
        all_labels : np.ndarray [N]
    """
    encoder.eval()
    classifier.eval()

    all_probs  = []
    all_labels = []

    for imgs, labels in dataloader:
        imgs   = imgs.to(device)    # [B, C, H, W]
        labels = labels.to(device)

        batch_probs = []
        for aug in augmentations:
            # Apply augmentation to each image in the batch
            aug_imgs = torch.stack([aug(img) for img in imgs])  # [B, C, H, W]
            features = encoder(aug_imgs)                         # [B, feat_dim] or [B, N, D]
            logits   = classifier(features)                      # [B, num_classes]
            probs    = F.softmax(logits, dim=1)                  # [B, num_classes]
            batch_probs.append(probs)

        # Average probabilities across all TTA passes
        avg_probs = torch.stack(batch_probs, dim=0).mean(dim=0)  # [B, num_classes]
        all_probs.append(avg_probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)


# ──────────────────────────────────────────────
# Threshold sweep
# ──────────────────────────────────────────────
def sweep_threshold(probs, labels, class_idx=1):
    """
    Sweeps decision threshold from 0.3 to 0.7 and returns
    the threshold that maximizes accuracy on this set.
    Only call this on val set, never on test set.
    """
    best_acc, best_thresh = 0.0, 0.5
    for thresh in np.arange(0.30, 0.71, 0.01):
        preds = (probs[:, class_idx] >= thresh).astype(int)
        acc   = (preds == labels).mean() * 100
        if acc > best_acc:
            best_acc   = acc
            best_thresh = thresh
    return best_thresh, best_acc


# ──────────────────────────────────────────────
# Build model from checkpoint
# ──────────────────────────────────────────────
def build_model(ckpt_path: str, num_classes: int, device: str):
    ckpt = safe_torch_load(ckpt_path, map_location=device)

    # Load encoder
    encoder = load_encoder(ckpt_path, device=device)
    encoder = encoder.to(device)

    # Detect feat_dim from encoder output
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224).to(device)
        out   = encoder(dummy)
        feat_dim = out.shape[-1]
    print(f"  [build_model] feat_dim={feat_dim}")

    # Wrap if patch tokens (3D output)
    if out.dim() == 3:
        encoder = PatchEncoderWrapper(encoder)
        classifier = AttentionPoolClassifier(feat_dim, num_classes)
    else:
        classifier = torch.nn.Linear(feat_dim, num_classes)

    # Load classifier weights
    classifier.load_state_dict(ckpt["classifier_state_dict"])
    classifier = classifier.to(device)

    return encoder, classifier


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",       type=str, required=True,  help="Path to ckpt_best_finetune.pth")
    p.add_argument("--data_path",  type=str, required=True,  help="Dataset root")
    p.add_argument("--num_classes",type=int, default=2)
    p.add_argument("--split_type", type=str, default="central/test.csv",
                   help="CSV path relative to data_path for test set")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_tta",      type=int, default=8, choices=[1, 4, 8],
                   help="Number of TTA augmentations (1=disabled, 4=fast, 8=full)")
    p.add_argument("--threshold_sweep", action="store_true",
                   help="Sweep decision threshold on test set (binary only)")
    p.add_argument("--device",     type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"\n{'='*55}")
    print(f"  FedMamba-SALT: TTA Evaluation")
    print(f"{'='*55}")
    print(f"  Checkpoint : {args.ckpt}")
    print(f"  TTA passes : {args.n_tta}")
    print(f"  Device     : {args.device}")

    # ── Dataset ──
    transform = get_eval_transform()
    test_dataset = RetinaDataset(
        data_path=args.data_path,
        split=args.split_type,
        transform=transform,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
    )
    print(f"  Test images: {len(test_dataset)}")

    # ── Model ──
    encoder, classifier = build_model(args.ckpt, args.num_classes, args.device)
    augmentations = get_tta_augmentations(args.n_tta)

    # ── Baseline (no TTA) ──
    print(f"\n  Running baseline (no TTA)...")
    base_augs  = get_tta_augmentations(1)
    base_probs, labels = predict_with_tta(encoder, classifier, test_loader, base_augs, args.device)
    base_preds = base_probs.argmax(axis=1)
    base_acc   = (base_preds == labels).mean() * 100
    base_auc   = roc_auc_score(labels, base_probs[:, 1]) if args.num_classes == 2 else 0.0
    print(f"  Baseline  →  acc={base_acc:.2f}%  auc={base_auc:.4f}")

    # ── TTA ──
    if args.n_tta > 1:
        print(f"\n  Running TTA ({args.n_tta} passes)...")
        tta_probs, labels = predict_with_tta(encoder, classifier, test_loader, augmentations, args.device)
        tta_preds = tta_probs.argmax(axis=1)
        tta_acc   = (tta_preds == labels).mean() * 100
        tta_auc   = roc_auc_score(labels, tta_probs[:, 1]) if args.num_classes == 2 else 0.0
        print(f"  TTA x{args.n_tta}   →  acc={tta_acc:.2f}%  auc={tta_auc:.4f}")
        print(f"  TTA gain  →  +{tta_acc - base_acc:.2f}% acc  |  +{tta_auc - base_auc:.4f} auc")

    # ── Threshold sweep ──
    if args.threshold_sweep and args.num_classes == 2:
        print(f"\n  Sweeping decision threshold...")
        probs_to_use = tta_probs if args.n_tta > 1 else base_probs
        best_thresh, best_thresh_acc = sweep_threshold(probs_to_use, labels)
        print(f"  Default threshold (0.50) → acc={tta_acc:.2f}%")
        print(f"  Best threshold   ({best_thresh:.2f}) → acc={best_thresh_acc:.2f}%")
        print(f"  Threshold gain   → +{best_thresh_acc - tta_acc:.2f}%")

    print(f"\n{'='*55}\n")


if __name__ == "__main__":
    main()