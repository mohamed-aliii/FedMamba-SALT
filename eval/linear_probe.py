"""
eval/linear_probe.py -- Linear probing & full fine-tuning evaluation.

Linear probing is the standard diagnostic for self-supervised representations:
freeze the encoder, attach a single nn.Linear classifier, train only that layer
on labeled data, and report test accuracy.  High accuracy means the encoder
learned semantically meaningful features without labels.

Supports two modes via --mode:
  linear_probe   (default) -- Encoder fully frozen, only the classifier trains.
  full_finetune  -- Encoder is unfrozen and fine-tuned end-to-end alongside
                    the classifier, which is the apples-to-apples comparison
                    with methods like Fed-MAE.

Usage:
    python -m eval.linear_probe \\
        --encoder_ckpt outputs/ckpt_latest.pth \\
        --data_path /path/to/dataset \\
        --num_classes 5 \\
        --mode linear_probe
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend before any pyplot import
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision import transforms
from tqdm import tqdm

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from augmentations.retina_dataset import RetinaDataset
from augmentations.medical_aug import RETINA_MEAN, RETINA_STD
from models.inception_mamba import InceptionMambaEncoder
from utils.ckpt_compat import safe_torch_load


# ======================================================================
# Constants
# ======================================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Fed-MAE baseline for context
FEDMAE_BASELINE = 77.43  # % accuracy, full fine-tuning, Retina Split-3


# ======================================================================
# CLI
# ======================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Linear probe / fine-tune evaluation for FedMamba-SALT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--encoder_ckpt", type=str, required=True,
                   help="Path to a checkpoint from train_centralized.py")
    p.add_argument("--data_path", type=str, required=True,
                   help="Dataset root (must contain train/ and test/ subdirs)")
    p.add_argument("--num_classes", type=int, required=True,
                   help="Number of classification classes")
    p.add_argument("--output_dir", type=str, default="eval_results",
                   help="Directory for confusion matrix PNG and logs")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--mode", type=str, default="linear_probe",
        choices=["linear_probe", "full_finetune"],
        help="linear_probe: freeze encoder; full_finetune: train encoder + classifier",
    )
    return p.parse_args()


# ======================================================================
# Transforms
# ======================================================================
def get_eval_transform(dataset: str = "retina") -> transforms.Compose:
    """
    Clean evaluation transform -- NO augmentations.
    Using heavy student augmentations here would artificially depress
    accuracy and make pre-training look worse than it is.
    """
    mean = RETINA_MEAN if dataset == "retina" else IMAGENET_MEAN
    std = RETINA_STD if dataset == "retina" else IMAGENET_STD
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_train_transform(dataset: str = "retina") -> transforms.Compose:
    """
    Mild training transform for full fine-tuning mode.
    Slightly stronger than eval but much weaker than the SALT student pipeline.
    """
    mean = RETINA_MEAN if dataset == "retina" else IMAGENET_MEAN
    std = RETINA_STD if dataset == "retina" else IMAGENET_STD
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# ======================================================================
# Encoder loading
# ======================================================================
def load_encoder(ckpt_path: str, device: str, freeze: bool) -> InceptionMambaEncoder:
    """
    Load the student encoder from a train_centralized.py checkpoint.

    Args:
        ckpt_path: Path to the checkpoint .pth file.
        device:    Target device.
        freeze:    If True, freeze all encoder parameters (linear probe mode).
    """
    encoder = InceptionMambaEncoder(
        patch_size=16, embed_dim=256, depth=4, out_dim=768,
    )

    ckpt = safe_torch_load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("student_state_dict", ckpt)
    encoder.load_state_dict(state_dict)
    print(f"[Encoder] Loaded weights from: {ckpt_path}")

    encoder = encoder.to(device)

    if freeze:
        for param in encoder.parameters():
            param.requires_grad = False
        encoder.eval()
        trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        assert trainable == 0, f"Encoder has {trainable} trainable params after freezing!"
        print(f"[Encoder] Frozen -- 0 trainable parameters")
    else:
        trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        print(f"[Encoder] Unfrozen -- {trainable / 1e6:.2f}M trainable parameters")

    return encoder


# ======================================================================
# Feature extraction (for linear probe mode)
# ======================================================================
@torch.no_grad()
def extract_features(
    encoder: nn.Module,
    dataloader: DataLoader,
    device: str,
) -> tuple:
    """
    Pre-compute all (feature, label) pairs from a DataLoader.

    This avoids running the encoder on every epoch and is standard practice
    (ref: DINO eval_linear.py at github.com/facebookresearch/dino).

    Returns:
        features: Tensor of shape (N, 768)
        labels:   Tensor of shape (N,)
    """
    encoder.eval()
    all_features = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc="  Extracting features", leave=False):
        images = images.to(device, non_blocking=True)
        # Extract and explicitly L2-normalize. SALT optimized cosine similarity,
        # so features MUST be on the unit sphere for linear probing to work effectively.
        features = encoder(images)  # (B, 768)
        features = F.normalize(features, dim=-1, p=2)
        all_features.append(features.cpu())
        all_labels.append(labels)

    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)


# ======================================================================
# Linear classifier training (on cached features)
# ======================================================================
def train_linear_classifier(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    num_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
) -> nn.Linear:
    """Train a single nn.Linear layer on cached features."""
    feat_dim = train_features.shape[1]
    classifier = nn.Linear(feat_dim, num_classes).to(device)
    nn.init.kaiming_uniform_(classifier.weight)
    nn.init.zeros_(classifier.bias)

    optimizer = Adam(classifier.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(train_features, train_labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    print(f"\n  Training linear classifier on {len(train_features)} cached features...")
    for epoch in range(epochs):
        classifier.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for feats, labels in loader:
            feats = feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = classifier(feats)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * feats.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += feats.size(0)

        avg_loss = total_loss / total
        acc = 100.0 * correct / total
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch [{epoch + 1:3d}/{epochs}]  "
                  f"loss={avg_loss:.4f}  train_acc={acc:.2f}%")

    return classifier


# ======================================================================
# Full fine-tuning training
# ======================================================================
def train_finetune(
    encoder: nn.Module,
    classifier: nn.Linear,
    train_loader: DataLoader,
    epochs: int,
    lr: float,
    device: str,
) -> None:
    """End-to-end fine-tuning of encoder + classifier."""
    params = list(encoder.parameters()) + list(classifier.parameters())
    optimizer = AdamW(params, lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    print(f"\n  Fine-tuning encoder + classifier on {len(train_loader.dataset)} images...")
    for epoch in range(epochs):
        encoder.train()
        classifier.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"  Epoch {epoch+1}", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            features = encoder(images)
            logits = classifier(features)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

        avg_loss = total_loss / total
        acc = 100.0 * correct / total
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch [{epoch + 1:3d}/{epochs}]  "
                  f"loss={avg_loss:.4f}  train_acc={acc:.2f}%")


# ======================================================================
# Evaluation
# ======================================================================
@torch.no_grad()
def evaluate(
    features: torch.Tensor,
    labels: torch.Tensor,
    classifier: nn.Linear,
    num_classes: int,
    device: str,
    class_names: list = None,
) -> tuple:
    """
    Evaluate on cached features.
    Returns (top1_accuracy, per_class_accuracy, confusion_matrix).
    """
    classifier.eval()
    feats = features.to(device)
    labs = labels.to(device)

    logits = classifier(feats)
    preds = logits.argmax(dim=1)

    # Top-1 accuracy
    correct = (preds == labs).sum().item()
    top1 = 100.0 * correct / len(labs)

    # Confusion matrix (C x C)
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(labs.cpu(), preds.cpu()):
        cm[t.item(), p.item()] += 1

    # Per-class accuracy
    per_class = {}
    for c in range(num_classes):
        total_c = cm[c].sum().item()
        correct_c = cm[c, c].item()
        name = class_names[c] if class_names else str(c)
        per_class[name] = 100.0 * correct_c / max(1, total_c)

    return top1, per_class, cm.numpy()


@torch.no_grad()
def evaluate_finetune(
    encoder: nn.Module,
    classifier: nn.Linear,
    dataloader: DataLoader,
    num_classes: int,
    device: str,
    class_names: list = None,
) -> tuple:
    """Evaluate fine-tuned encoder + classifier on a DataLoader."""
    encoder.eval()
    classifier.eval()

    all_preds = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc="  Evaluating", leave=False):
        images = images.to(device, non_blocking=True)
        features = encoder(images)
        logits = classifier(features)
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(labels)

    preds = torch.cat(all_preds)
    labs = torch.cat(all_labels)

    correct = (preds == labs).sum().item()
    top1 = 100.0 * correct / len(labs)

    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(labs, preds):
        cm[t.item(), p.item()] += 1

    per_class = {}
    for c in range(num_classes):
        total_c = cm[c].sum().item()
        correct_c = cm[c, c].item()
        name = class_names[c] if class_names else str(c)
        per_class[name] = 100.0 * correct_c / max(1, total_c)

    return top1, per_class, cm.numpy()


# ======================================================================
# Visualization
# ======================================================================
def save_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    output_path: str,
    title: str = "Confusion Matrix",
) -> None:
    """Save a confusion matrix heatmap as PNG."""
    num_classes = len(class_names)
    fig, ax = plt.subplots(figsize=(max(6, num_classes), max(5, num_classes - 1)))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=range(num_classes),
        yticks=range(num_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label",
        xlabel="Predicted Label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells with counts
    thresh = cm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved to: {output_path}")


# ======================================================================
# Main
# ======================================================================
def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    mode_label = "Linear Probe" if args.mode == "linear_probe" else "Full Fine-tune"
    print("=" * 60)
    print(f"  FedMamba-SALT Evaluation -- {mode_label}")
    print("=" * 60)

    # -------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------
    eval_transform = get_eval_transform(dataset="retina")
    train_transform = get_train_transform(dataset="retina") if args.mode == "full_finetune" else eval_transform

    train_ds = RetinaDataset(
        data_path=args.data_path,
        phase="train",
        split_type="central",
        split_csv="train.csv",
        transform=train_transform,
    )
    test_ds = RetinaDataset(
        data_path=args.data_path,
        phase="test",
        split_type="central",
        split_csv="test.csv",
        transform=eval_transform,
    )
    class_names = train_ds.classes

    print(f"  Train: {len(train_ds)} images, {len(class_names)} classes")
    print(f"  Test:  {len(test_ds)} images")
    print(f"  Classes: {class_names}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # -------------------------------------------------------------------
    # Encoder
    # -------------------------------------------------------------------
    freeze = (args.mode == "linear_probe")
    encoder = load_encoder(args.encoder_ckpt, args.device, freeze=freeze)

    # -------------------------------------------------------------------
    # Classifier
    # -------------------------------------------------------------------
    classifier = nn.Linear(768, args.num_classes).to(args.device)
    nn.init.kaiming_uniform_(classifier.weight)
    nn.init.zeros_(classifier.bias)

    # -------------------------------------------------------------------
    # Train + Evaluate
    # -------------------------------------------------------------------
    if args.mode == "linear_probe":
        # Pre-compute features for efficiency (DINO-style)
        print("\n  [Phase 1] Extracting train features...")
        train_feats, train_labels = extract_features(encoder, train_loader, args.device)
        print(f"    Cached {train_feats.shape[0]} features of dim {train_feats.shape[1]}")

        print("  [Phase 2] Extracting test features...")
        test_feats, test_labels = extract_features(encoder, test_loader, args.device)
        print(f"    Cached {test_feats.shape[0]} features of dim {test_feats.shape[1]}")

        # Train linear classifier on cached features
        classifier = train_linear_classifier(
            train_feats, train_labels,
            num_classes=args.num_classes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
        )

        # Evaluate on cached test features
        top1, per_class, cm = evaluate(
            test_feats, test_labels, classifier,
            args.num_classes, args.device, class_names,
        )

    else:  # full_finetune
        # Train end-to-end
        train_finetune(
            encoder, classifier, train_loader,
            epochs=args.epochs, lr=args.lr, device=args.device,
        )

        # Evaluate
        top1, per_class, cm = evaluate_finetune(
            encoder, classifier, test_loader,
            args.num_classes, args.device, class_names,
        )

    # -------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"  {mode_label} Results")
    print("=" * 60)
    print(f"\n  Top-1 Accuracy: {top1:.2f}%\n")
    print("  Per-class accuracy:")
    for name, acc in per_class.items():
        print(f"    {name:>20s}: {acc:.2f}%")

    # Save confusion matrix
    cm_path = os.path.join(args.output_dir, f"confusion_matrix_{args.mode}.png")
    save_confusion_matrix(cm, class_names, cm_path, title=f"{mode_label} Confusion Matrix")

    # Final summary line
    print(f"\n  {mode_label} Result: {top1:.2f}% on {args.num_classes} classes "
          f"over {len(test_ds)} test samples")

    # -------------------------------------------------------------------
    # Baseline comparison reminder
    # -------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("  BASELINE COMPARISON NOTE:")
    print(f"    Fed-MAE baseline on Retina Split-3: ~{FEDMAE_BASELINE:.2f}%")
    print("    (full fine-tuning, NOT linear probing)")
    print()
    if args.mode == "linear_probe":
        print("    Linear probing is a DIAGNOSTIC -- it measures representation")
        print("    quality but is NOT an apples-to-apples comparison with Fed-MAE.")
        print("    For a fair comparison, also run with --mode full_finetune.")
    else:
        print(f"    Your full fine-tune result: {top1:.2f}% vs Fed-MAE: {FEDMAE_BASELINE:.2f}%")
        diff = top1 - FEDMAE_BASELINE
        sign = "+" if diff >= 0 else ""
        print(f"    Delta: {sign}{diff:.2f}%")
    print("-" * 60)


if __name__ == "__main__":
    main()
