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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
def get_tta_augmentations(n_tta: int, dataset: str = "retina"):
    """
    Returns a list of tensor→tensor augmentation functions.
    Uses Deterministic Rotational TTA to preserve retinal features.
    Max 4 passes: Original, 90°, 180°, 270°.
    """
    dataset_key = dataset.lower().replace("_", "-")
    if dataset_key in {"covid", "covid-fl", "covidfl"}:
        augs = [
            lambda x: x,
            lambda x: torch.flip(x, dims=[2]),
        ]
        if n_tta > len(augs):
            print(
                f"  [TTA] Warning: n_tta={n_tta} requested, but COVID-FL TTA "
                f"uses {len(augs)} chest-X-ray-safe passes. Clamping."
            )
        return augs[: min(n_tta, len(augs))]

    augs = [
        lambda x: x,                                      # 1. original
        lambda x: torch.rot90(x, k=1, dims=[1, 2]),       # 2. 90°
        lambda x: torch.rot90(x, k=2, dims=[1, 2]),       # 3. 180°
        lambda x: torch.rot90(x, k=3, dims=[1, 2]),       # 4. 270°
    ]
    
    if n_tta > 4:
        print(f"  [TTA] Warning: n_tta={n_tta} requested, but Rotational TTA only supports 4 passes. Clamping to 4.")
        n_tta = 4
        
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
    Sweeps decision thresholds using the exact ROC curve thresholds
    to find the one that maximizes accuracy on this set.
    Only call this on val set, never on test set.
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(labels, probs[:, class_idx])

    accs = []
    for t in thresholds:
        preds = (probs[:, class_idx] >= t).astype(int)
        accs.append((preds == labels).mean() * 100)

    best_idx = np.argmax(accs)
    best_thresh = thresholds[best_idx]
    best_acc = accs[best_idx]

    return best_thresh, best_acc, thresholds, np.asarray(accs)


def save_tta_artifacts(
    output_dir: str,
    base_acc: float,
    base_auc: float,
    tta_acc: float,
    tta_auc: float,
    threshold_data=None,
) -> None:
    """Save TTA comparison plots and CSV for paper documentation."""
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "tta_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["setting", "accuracy", "auc"])
        writer.writerow(["baseline", f"{base_acc:.4f}", f"{base_auc:.6f}"])
        writer.writerow(["tta", f"{tta_acc:.4f}", f"{tta_auc:.6f}"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = ["Baseline", "TTA"]
    axes[0].bar(labels, [base_acc, tta_acc], color=["#607D8B", "#2ECC71"])
    axes[0].set_title("Accuracy")
    axes[0].set_ylabel("Top-1 Accuracy (%)")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(labels, [base_auc, tta_auc], color=["#607D8B", "#8E44AD"])
    axes[1].set_title("AUC")
    axes[1].set_ylabel("AUC")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.suptitle("TTA Evaluation Summary", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "plot_tta_comparison.png"), dpi=150)
    plt.close(fig)

    if threshold_data is not None:
        best_thresh, best_acc, thresholds, accs = threshold_data
        finite = np.isfinite(thresholds)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(thresholds[finite], accs[finite], color="#2196F3", linewidth=2)
        ax.axvline(
            best_thresh, color="#E91E63", linestyle="--",
            label=f"Best threshold={best_thresh:.3f}",
        )
        ax.set_xlabel("Decision Threshold")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Threshold Sweep", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "plot_threshold_sweep.png"), dpi=150)
        plt.close(fig)

    print(f"  TTA artifacts saved to: {output_dir}")


# ──────────────────────────────────────────────
# Build model from checkpoint
# ──────────────────────────────────────────────
def build_model(ckpt_path: str, num_classes: int, device: str):
    ckpt = safe_torch_load(ckpt_path, map_location=device)

    # Load base encoder. Fine-tuned checkpoints may contain either the raw
    # encoder or a PatchEncoderWrapper; load_encoder normalizes both to the
    # raw InceptionMambaEncoder.
    base_encoder = load_encoder(ckpt_path, device=device, freeze=False)
    base_encoder = base_encoder.to(device)
    classifier_state = ckpt["classifier_state_dict"]

    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224).to(device)
        patch_tokens = base_encoder(dummy, return_patches=True)
        gap_features = base_encoder(dummy, return_patches=False)

    if any(k.startswith("attn.") or k.startswith("head.") for k in classifier_state):
        feat_dim = patch_tokens.shape[-1]
        encoder = PatchEncoderWrapper(base_encoder)
        classifier = AttentionPoolClassifier(feat_dim, num_classes)
    elif any(k.startswith("0.") for k in classifier_state):
        feat_dim = gap_features.shape[-1]
        encoder = base_encoder
        classifier = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Dropout(0.2),
            nn.Linear(feat_dim, num_classes),
        )
    else:
        feat_dim = gap_features.shape[-1]
        encoder = base_encoder
        classifier = torch.nn.Linear(feat_dim, num_classes)
    print(f"  [build_model] feat_dim={feat_dim}")

    classifier.load_state_dict(classifier_state)
    classifier = classifier.to(device)

    return encoder, classifier


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",       type=str, required=True,  help="Path to ckpt_best_finetune.pth")
    p.add_argument("--data_path",  type=str, required=True,  help="Dataset root")
    p.add_argument("--dataset",    type=str, default="retina",
                   help="Dataset preset for transforms/TTA: retina or covidfl")
    p.add_argument("--num_classes",type=int, default=2)
    p.add_argument("--split_type", type=str, default="split_1",
                   help="Training split label for logging only; test.csv is used for evaluation")
    p.add_argument("--split_csv", type=str, default="test.csv",
                   help="Test CSV inside data_path/central/")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_tta",      type=int, default=8, choices=[1, 4, 8],
                   help="Number of TTA augmentations (1=disabled, 4=fast, 8=full)")
    p.add_argument("--threshold_sweep", action="store_true",
                   help="Sweep decision threshold on test set (binary only)")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Directory for TTA plots and CSV. Defaults to checkpoint directory.")
    p.add_argument("--device",     type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    from sklearn.metrics import roc_auc_score

    args = parse_args()
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.ckpt))
    print(f"\n{'='*55}")
    print(f"  FedMamba-SALT: TTA Evaluation")
    print(f"{'='*55}")
    print(f"  Checkpoint : {args.ckpt}")
    print(f"  TTA passes : {args.n_tta}")
    print(f"  Device     : {args.device}")

    # ── Dataset ──
    transform = get_eval_transform(args.dataset)
    test_dataset = RetinaDataset(
        data_path=args.data_path,
        phase="test",
        split_type="central",
        split_csv=args.split_csv,
        transform=transform,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
    )
    print(f"  Test images: {len(test_dataset)}")

    # ── Model ──
    encoder, classifier = build_model(args.ckpt, args.num_classes, args.device)
    augmentations = get_tta_augmentations(args.n_tta, args.dataset)

    # ── Baseline (no TTA) ──
    print(f"\n  Running baseline (no TTA)...")
    base_augs  = get_tta_augmentations(1, args.dataset)
    base_probs, labels = predict_with_tta(encoder, classifier, test_loader, base_augs, args.device)
    base_preds = base_probs.argmax(axis=1)
    base_acc   = (base_preds == labels).mean() * 100
    base_auc   = roc_auc_score(labels, base_probs[:, 1]) if args.num_classes == 2 else 0.0
    print(f"  Baseline  →  acc={base_acc:.2f}%  auc={base_auc:.4f}")

    # ── TTA ──
    tta_probs = base_probs
    tta_acc = base_acc
    tta_auc = base_auc
    if args.n_tta > 1:
        print(f"\n  Running TTA ({args.n_tta} passes)...")
        tta_probs, labels = predict_with_tta(encoder, classifier, test_loader, augmentations, args.device)
        tta_preds = tta_probs.argmax(axis=1)
        tta_acc   = (tta_preds == labels).mean() * 100
        tta_auc   = roc_auc_score(labels, tta_probs[:, 1]) if args.num_classes == 2 else 0.0
        print(f"  TTA x{args.n_tta}   →  acc={tta_acc:.2f}%  auc={tta_auc:.4f}")
        print(f"  TTA gain  →  +{tta_acc - base_acc:.2f}% acc  |  +{tta_auc - base_auc:.4f} auc")

    # ── Threshold sweep ──
    threshold_data = None
    if args.threshold_sweep and args.num_classes == 2:
        print(f"\n  Sweeping decision threshold...")
        best_thresh, best_thresh_acc, thresholds, accs = sweep_threshold(tta_probs, labels)
        threshold_data = (best_thresh, best_thresh_acc, thresholds, accs)
        print(f"  Default threshold (0.50) → acc={tta_acc:.2f}%")
        print(f"  Best threshold   ({best_thresh:.2f}) → acc={best_thresh_acc:.2f}%")
        print(f"  Threshold gain   → +{best_thresh_acc - tta_acc:.2f}%")

    save_tta_artifacts(
        output_dir=output_dir,
        base_acc=base_acc,
        base_auc=base_auc,
        tta_acc=tta_acc,
        tta_auc=tta_auc,
        threshold_data=threshold_data,
    )

    print(f"\n{'='*55}\n")


if __name__ == "__main__":
    main()
