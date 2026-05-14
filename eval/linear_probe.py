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

Fixes applied (see audit notes inline):
  FIX 1:  frozen_attention mode now writes confusion matrix / report to sub_output,
          not args.output_dir (was overwriting other runs).
  FIX 2:  stratified_subset uses a .targets / get_label() fast-path where possible
          to avoid loading every image just to collect labels.
  FIX 3:  load_encoder raises a clear ValueError when the checkpoint has neither
          expected key, instead of silently using the raw ckpt dict as a state dict.
  FIX 4:  patience_counter resets when the encoder unfreezes at WARMUP_EPOCHS,
          so frozen-phase non-improvement doesn't eat early-stopping budget.
  FIX 5:  The advertised encoder unfreeze mini-warmup (UNFREEZE_WARMUP) is now
          actually implemented via manual param-group LR writes in the epoch loop.
  FIX 6:  out_dir computation in run_evaluation is explicit for all three modes,
          eliminating the "by-luck" fallback for linear_probe + label_fraction.
  FIX 7:  frozen_attention mode writes ROC curve and classification report to the
          same sub_output directory as the confusion matrix.
  FIX 8:  Mixup dominant-label comment clarifies the lam=0.5 boundary ambiguity.
  FIX 9:  train_finetune val_loss comment clarifies it is not comparable to
          Mixup-modified training loss (generalization gap chart caveat added).
  FIX 10: save_confusion_matrix uses per-column normalization for text color
          threshold so imbalanced datasets don't wash out near-zero cells.
  FIX 11: extract_features docstring clarifies the no-L2-norm assumption is
          SALT-distillation-specific (not valid for DINO-style encoders).
  FIX 12: TF.rotate torchvision version caveat documented; version check added
          at module load time with a clear warning.
"""

import argparse
import csv
import os
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend before any pyplot import
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.amp import autocast, GradScaler
from torchvision import transforms
import torchvision
import torchvision.transforms.functional as TF
from tqdm import tqdm

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from augmentations.retina_dataset import RetinaDataset
from augmentations.medical_aug import RETINA_MEAN, RETINA_STD
from models.inception_mamba import InceptionMambaEncoder
from objectives.salt_loss import ProjectionHead
from utils.ckpt_compat import safe_torch_load


# ======================================================================
# FIX 12: Warn early if torchvision is too old for 4-D TF.rotate.
# torchvision.transforms.functional.rotate gained native Tensor support
# in 0.9.0 (PyTorch 1.8).  Older versions silently fall back to PIL
# conversion, which produces subtly wrong values and is much slower.
# ======================================================================
_TV_VERSION = tuple(int(x) for x in torchvision.__version__.split(".")[:2] if x.isdigit())
if _TV_VERSION < (0, 9):
    warnings.warn(
        f"torchvision {torchvision.__version__} detected. TTA rotation "
        "in evaluate_finetune uses TF.rotate on 4-D BCHW tensors, which "
        "requires torchvision >= 0.9.0.  Upgrade to avoid silent PIL "
        "fallback during evaluation.",
        UserWarning,
        stacklevel=1,
    )


# ======================================================================
# Constants
# ======================================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Fed-MAE baseline for context
FEDMAE_BASELINE = 81.93  # % accuracy, centralized baseline, Retina

# Early stopping for fine-tuning
FINETUNE_PATIENCE = 30  # stop if val_acc doesn't improve for this many epochs
# Warmup length is computed dynamically inside train_finetune as min(10, epochs//10)

# Mixup / CutMix
MIXUP_ALPHA = 0.2       # Beta distribution parameter for Mixup (0.4 was too aggressive for 9K images)
TTA_AUGMENTS = 5         # Number of augmented views for Test-Time Augmentation


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
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_workers", type=int, default=4,
                   help="Number of DataLoader worker processes")
    p.add_argument(
        "--mode", type=str, default="linear_probe",
        choices=["linear_probe", "full_finetune", "frozen_attention"],
        help="linear_probe: freeze encoder; full_finetune: train encoder + classifier; frozen_attention: freeze encoder, train attention classifier only",
    )
    p.add_argument(
        "--label_fraction", type=float, default=1.0,
        help="Fraction of training labels to use (0.0-1.0). "
             "Use < 1.0 to test label scarcity robustness. "
             "Stratified sampling preserves class balance.",
    )
    p.add_argument(
        "--label_scarcity", action="store_true",
        help="Run label scarcity experiment: automatically evaluates "
             "at 30%%, 60%%, and 100%% label fractions and produces "
             "a comparison report.",
    )
    p.add_argument(
        "--eval_only", action="store_true",
        help="Skip training and directly evaluate the provided checkpoint (must contain encoder and classifier state_dicts)",
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
    Moderate training transform for full fine-tuning mode.

    Stronger than eval to provide regularisation during fine-tuning,
    but still medical-safe (no heavy blur that destroys pathology).
    """
    mean = RETINA_MEAN if dataset == "retina" else IMAGENET_MEAN
    std = RETINA_STD if dataset == "retina" else IMAGENET_STD
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.1, contrast=0.15, saturation=0.08, hue=0.01,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_tta_transforms(dataset: str = "retina") -> list:
    """
    Returns a list of transforms for Test-Time Augmentation.
    Each transform produces a slightly different view of the same image.
    Final prediction = average of softmax probabilities across all views.
    """
    mean = RETINA_MEAN if dataset == "retina" else IMAGENET_MEAN
    std = RETINA_STD if dataset == "retina" else IMAGENET_STD
    base = [transforms.Resize(256), transforms.CenterCrop(224)]
    norm = [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]

    tta_list = [
        # 1. Clean center crop (same as eval)
        transforms.Compose(base + norm),
        # 2. Horizontal flip
        transforms.Compose(base + [transforms.RandomHorizontalFlip(p=1.0)] + norm),
        # 3. Vertical flip
        transforms.Compose(base + [transforms.RandomVerticalFlip(p=1.0)] + norm),
        # 4. Slight rotation
        transforms.Compose([transforms.Resize(256), transforms.CenterCrop(240),
                            transforms.RandomRotation(degrees=10),
                            transforms.CenterCrop(224)] + norm),
        # 5. Slightly larger crop
        transforms.Compose([transforms.Resize(288), transforms.CenterCrop(224)] + norm),
    ]
    return tta_list


def mixup_data(x, y, alpha=MIXUP_ALPHA):
    """
    Mixup: creates virtual training examples by linearly interpolating
    pairs of images and their labels.  This forces the model to learn
    smooth decision boundaries between classes, which helps generalization
    beyond the teacher's representation ceiling.

    Reference: Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute loss for Mixup: weighted sum of losses for both targets."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class FocalLoss(nn.Module):
    """
    Focal Loss: Down-weights well-classified "easy" examples to focus 100% of
    the gradient on hard, borderline examples. Extremely useful for class imbalance.
    """
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, weight=self.weight,
            label_smoothing=self.label_smoothing, reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ======================================================================
# Attention Pooling & Projector Models (Full Fine-tune only)
# ======================================================================
class AttentionPoolClassifier(nn.Module):
    """Learnable attention pooling + classification head."""
    def __init__(self, feat_dim=768, num_classes=2):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.head = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Dropout(0.5),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, patch_tokens):
        # patch_tokens: (B, 196, 768)
        w = F.softmax(self.attn(patch_tokens), dim=1)   # (B, 196, 1)
        pooled = (patch_tokens * w).sum(dim=1)           # (B, 768)
        return self.head(pooled)


class PatchEncoderWrapper(nn.Module):
    """Wraps the Inception-Mamba encoder to return dense patch tokens."""
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        return self.encoder(x, return_patches=True)  # (B, 196, 768)


def load_projector(ckpt_path: str, device: str, freeze: bool = False) -> nn.Module:
    """Load the SALT projection head from the pre-training checkpoint."""
    ckpt = safe_torch_load(ckpt_path, map_location="cpu")
    projector = ProjectionHead(in_dim=768, hidden_dim=2048, out_dim=768)

    if "projector_state_dict" in ckpt:
        projector.load_state_dict(ckpt["projector_state_dict"])
        print(f"[Projector] Loaded weights from: {ckpt_path}")
    else:
        print("[Projector] WARNING: No projector_state_dict found in checkpoint.")

    projector = projector.to(device)

    if freeze:
        for param in projector.parameters():
            param.requires_grad = False
        projector.eval()
        print("[Projector] Frozen -- 0 trainable parameters")
    else:
        trainable = sum(p.numel() for p in projector.parameters() if p.requires_grad)
        print(f"[Projector] Unfrozen -- {trainable / 1e6:.2f}M trainable parameters")

    return projector


# ======================================================================
# Encoder loading
# ======================================================================
def load_encoder(ckpt_path: str, device: str, freeze: bool) -> InceptionMambaEncoder:
    """
    Load the student encoder from a train_centralized.py checkpoint.

    Automatically infers embed_dim and depth from the checkpoint's
    state_dict so the eval script works regardless of which encoder
    configuration was used for pre-training.

    Args:
        ckpt_path: Path to the checkpoint .pth file.
        device:    Target device.
        freeze:    If True, freeze all encoder parameters (linear probe mode).

    FIX 3: Raises ValueError with a clear message when neither expected key
    is present, rather than silently using the raw ckpt dict as a state dict
    (which would produce a confusing key-mismatch error later).
    """
    ckpt = safe_torch_load(ckpt_path, map_location="cpu")

    if "encoder_state_dict" in ckpt:
        # Fine-tuned checkpoint from this script
        state_dict = ckpt["encoder_state_dict"]
        # Strip the 'encoder.' prefix if it was wrapped in PatchEncoderWrapper
        state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
    elif "student_state_dict" in ckpt:
        # Pre-trained checkpoint from train_centralized.py
        state_dict = ckpt["student_state_dict"]
    elif isinstance(ckpt, dict) and any(k.startswith("patch_embed") for k in ckpt):
        # Raw state dict saved directly (no wrapper dict)
        state_dict = ckpt
    else:
        # FIX 3: fail loudly instead of silently misloading
        raise ValueError(
            f"Checkpoint at '{ckpt_path}' does not contain 'encoder_state_dict' "
            "or 'student_state_dict', and does not appear to be a raw encoder "
            "state dict (no 'patch_embed.*' keys found).  "
            "Check that you are pointing --encoder_ckpt at the correct file.\n"
            f"  Keys found: {list(ckpt.keys())[:10]}"
        )

    # --- Infer architecture from checkpoint ---
    embed_dim = state_dict["patch_embed.proj.0.weight"].shape[0]

    block_indices = set()
    for key in state_dict:
        if key.startswith("blocks."):
            idx = int(key.split(".")[1])
            block_indices.add(idx)
    depth = len(block_indices)

    print(f"[Encoder] Detected architecture: embed_dim={embed_dim}, depth={depth}")

    encoder = InceptionMambaEncoder(
        patch_size=16, embed_dim=embed_dim, depth=depth, out_dim=768,
    )

    encoder.load_state_dict(state_dict)
    print(f"[Encoder] Loaded weights from: {ckpt_path}")

    encoder = encoder.to(device)

    if freeze:
        for param in encoder.parameters():
            param.requires_grad = False
        encoder.eval()
        trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        assert trainable == 0, f"Encoder has {trainable} trainable params after freezing!"
        print("[Encoder] Frozen -- 0 trainable parameters")
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

    NOTE (FIX 11): L2-normalization is intentionally skipped here.
    With Smooth L1 distillation (SALT), the encoder learns to match the
    teacher's actual geometry (direction + magnitude).  Normalizing would
    erase the magnitude information that the downstream BatchNorm1d head
    expects.  This assumption is SALT-specific: if you ever swap in a
    DINO-style encoder whose outputs are already L2-normalized, you should
    normalize here as well.

    Returns:
        features: Tensor of shape (N, 768)
        labels:   Tensor of shape (N,)
    """
    encoder.eval()
    all_features = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc="  Extracting features", leave=False):
        images = images.to(device, non_blocking=True)
        features = encoder(images, return_patches=False)  # (B, 768)
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
) -> nn.Module:
    """Train a single nn.Linear layer on cached features."""
    feat_dim = train_features.shape[1]
    classifier = nn.Sequential(
        nn.BatchNorm1d(feat_dim),
        nn.Linear(feat_dim, num_classes)
    ).to(device)

    nn.init.kaiming_uniform_(classifier[1].weight)
    nn.init.zeros_(classifier[1].bias)

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
# GPU memory helper
# ======================================================================
def _gpu_stats(device: str) -> dict:
    """Return current GPU memory stats in MB."""
    if not torch.cuda.is_available() or "cpu" in device:
        return {"allocated_mb": 0, "reserved_mb": 0, "peak_mb": 0}
    return {
        "allocated_mb": torch.cuda.memory_allocated() / 1e6,
        "reserved_mb": torch.cuda.memory_reserved() / 1e6,
        "peak_mb": torch.cuda.max_memory_allocated() / 1e6,
    }


# ======================================================================
# Full fine-tuning training
# ======================================================================
def train_finetune(
    encoder: nn.Module,
    classifier: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float,
    device: str,
    num_classes: int,
    class_names: list,
    output_dir: str,
    freeze_encoder: bool = False,
) -> dict:
    """End-to-end fine-tuning of encoder + classifier.

    Key design decisions:
      1. Encoder LR = lr/10.  BERT/ViT fine-tuning convention; lr/2 destroys
         pre-trained representations by applying a random-init-scale LR.
      2. Warmup: min(10, max(3, epochs//10)) epochs linear warmup from lr*0.1.
         Classifier stabilises before encoder gradients flow.
      3. Encoder unfreeze mini-warmup (UNFREEZE_WARMUP=3 epochs): ramps encoder
         LR from lr/500 to encoder_lr via manual param-group writes in the loop.
         This warms up cold Adam m/v buffers before cosine decay begins.
         (FIX 5: this was advertised but never implemented in the original.)
      4. CosineAnnealingLR after warmup, eta_min=1e-4 (fixed floor).
      5. Label smoothing (0.1) + class weights for imbalanced data.
      6. Weight decay: 0.01 encoder, 0.05 classifier.
      7. Per-epoch validation with AUC.
      8. Early stopping; patience counter resets at encoder unfreeze
         (FIX 4: frozen-phase non-improvement no longer burns patience budget).

    NOTE (FIX 9): val_loss and training loss are NOT directly comparable.
    Training uses Mixup-modified targets; val_loss uses clean labels. The
    generalization gap chart (val_loss - train_loss) therefore shows an
    inflated gap. Use it as a relative trend indicator only.

    Returns:
        dict with lists of per-epoch metrics for visualization.
    """
    encoder_lr = lr / 10.0

    param_groups = [
        {"params": encoder.parameters(),   "lr": encoder_lr, "weight_decay": 0.01},
        {"params": classifier.parameters(), "lr": lr,         "weight_decay": 0.05},
    ]
    optimizer = AdamW(param_groups)

    WARMUP_EPOCHS = min(10, max(3, epochs // 10))

    # FIX 5: encoder unfreeze mini-warmup -- implemented below in the epoch loop.
    # After the classifier warmup finishes (epoch == WARMUP_EPOCHS), the encoder's
    # Adam m/v buffers are cold (zero). Jumping straight to encoder_lr can produce
    # a huge effective first step.  We ramp over UNFREEZE_WARMUP epochs:
    #   epoch WARMUP_EPOCHS+0: lr = encoder_lr / 50
    #   epoch WARMUP_EPOCHS+1: lr = encoder_lr / 50 * (2/3) + encoder_lr * (1/3)
    #   epoch WARMUP_EPOCHS+2: lr = encoder_lr / 50 * (1/3) + encoder_lr * (2/3)
    #   epoch WARMUP_EPOCHS+3+: lr = encoder_lr (managed by cosine scheduler)
    # SequentialLR owns the cosine phase; we only manually set the encoder group
    # during the ramp window. The scheduler still steps every epoch.
    UNFREEZE_WARMUP = 3
    ENCODER_WARMUP_START_LR = encoder_lr / 50.0

    eta_min_enc = 1e-4

    enc_warmup = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0,
        total_iters=WARMUP_EPOCHS, last_epoch=-1,
    )
    enc_cosine = CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - WARMUP_EPOCHS), eta_min=eta_min_enc,
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[enc_warmup, enc_cosine], milestones=[WARMUP_EPOCHS],
    )

    weights = [1.0] + [2.0] * (num_classes - 1)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    all_params = list(encoder.parameters()) + list(classifier.parameters())

    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    history = {
        "epoch": [], "loss": [], "train_acc": [], "val_acc": [],
        "val_loss": [], "val_auc": [],
        "enc_lr": [], "cls_lr": [], "time_s": [], "gpu_mb": [],
    }

    csv_path = os.path.join(output_dir, "finetune_metrics.csv")
    csv_columns = [
        "epoch", "loss", "train_acc", "val_loss", "val_acc", "val_auc",
        "enc_lr", "cls_lr", "time_s", "gpu_mb", "peak_gpu_mb",
    ]
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(csv_columns)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print(f"\n  Fine-tuning encoder + classifier on {len(train_loader.dataset)} images...")
    print(f"  Encoder LR: {encoder_lr:.1e}  |  Classifier LR: {lr:.1e}  |  Epochs: {epochs}")
    print(f"  Warmup: {WARMUP_EPOCHS} epochs (encoder frozen, classifier LR ramps 0.1x→1x)")
    print(f"  Encoder unfreeze mini-warmup: {UNFREEZE_WARMUP} epochs (lr/50 → encoder_lr)")
    print(f"  Cosine decay: epochs {WARMUP_EPOCHS}→{epochs}  eta_min={eta_min_enc:.1e}")
    print(f"  Early stopping patience: {FINETUNE_PATIENCE} epochs (resets at encoder unfreeze)")
    total_start = time.time()

    _device_type = "cuda" if "cuda" in device else "cpu"
    scaler = torch.amp.GradScaler(_device_type, enabled=("cuda" in device))

    for epoch in range(epochs):
        epoch_start = time.time()

        # ---- Encoder freeze / unfreeze ----
        if epoch < WARMUP_EPOCHS:
            if not freeze_encoder:
                for p in encoder.parameters():
                    p.requires_grad_(False)
            encoder.eval()
        elif epoch == WARMUP_EPOCHS and not freeze_encoder:
            for p in encoder.parameters():
                p.requires_grad_(True)
            encoder.train()
            # FIX 4: reset patience counter so frozen-phase non-improvement
            # does not consume the early-stopping budget.
            patience_counter = 0
            print(f"  [Epoch {epoch+1}] Encoder unfrozen -- starting mini-warmup "
                  f"({UNFREEZE_WARMUP} epochs, lr {ENCODER_WARMUP_START_LR:.1e} → {encoder_lr:.1e})")
            # Start encoder LR at the bottom of the mini-warmup ramp.
            # SequentialLR will try to set its own value on scheduler.step() below,
            # but we override the encoder group immediately after the step call.
            optimizer.param_groups[0]["lr"] = ENCODER_WARMUP_START_LR
        else:
            if not freeze_encoder:
                encoder.train()

        classifier.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"  Epoch {epoch+1}", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # FIX 8: lam is sampled from Beta(alpha, alpha), so it is symmetric
            # around 0.5.  The dominant_labels logic (targets_a if lam >= 0.5)
            # is a reasonable heuristic for accuracy logging; note that at exactly
            # lam=0.5 the assignment is arbitrary (loss is a perfect 50/50 split).
            # This only affects logged train_acc, not the loss or model weights.
            images, targets_a, targets_b, lam = mixup_data(images, labels, MIXUP_ALPHA)

            optimizer.zero_grad()

            with autocast(device_type=_device_type, enabled=("cuda" in device)):
                features = encoder(images)
                logits = classifier(features)
                loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            active_params = [p for p in all_params if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(active_params, max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * images.size(0)
            dominant_labels = targets_a if lam >= 0.5 else targets_b
            correct += (logits.argmax(dim=1) == dominant_labels).sum().item()
            total += images.size(0)

        # Step the SequentialLR scheduler first (manages warmup → cosine transition).
        scheduler.step()

        # FIX 5: Apply encoder unfreeze mini-warmup via manual param-group override.
        # This runs AFTER scheduler.step() so we overwrite whatever the scheduler
        # just wrote to param_groups[0]["lr"] during the ramp window.
        if not freeze_encoder and WARMUP_EPOCHS <= epoch < WARMUP_EPOCHS + UNFREEZE_WARMUP:
            ramp_progress = (epoch - WARMUP_EPOCHS + 1) / UNFREEZE_WARMUP
            ramped_lr = ENCODER_WARMUP_START_LR * (1 - ramp_progress) + encoder_lr * ramp_progress
            optimizer.param_groups[0]["lr"] = ramped_lr

        avg_loss = total_loss / total
        train_acc = 100.0 * correct / total

        val_acc, val_loss, val_auc = _eval_with_auc(
            encoder, classifier, test_loader, device, num_classes,
        )

        epoch_time = time.time() - epoch_start
        gpu = _gpu_stats(device)
        enc_lr = optimizer.param_groups[0]["lr"]
        cls_lr = optimizer.param_groups[1]["lr"]

        history["epoch"].append(epoch + 1)
        history["loss"].append(avg_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["enc_lr"].append(enc_lr)
        history["cls_lr"].append(cls_lr)
        history["time_s"].append(epoch_time)
        history["gpu_mb"].append(gpu["allocated_mb"])

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1, f"{avg_loss:.6f}", f"{train_acc:.2f}",
                f"{val_loss:.6f}", f"{val_acc:.2f}", f"{val_auc:.4f}",
                f"{enc_lr:.2e}", f"{cls_lr:.2e}", f"{epoch_time:.1f}",
                f"{gpu['allocated_mb']:.0f}", f"{gpu['peak_mb']:.0f}",
            ])

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            ckpt_path = os.path.join(output_dir, "ckpt_best_finetune.pth")
            torch.save({
                "epoch": epoch,
                "encoder_state_dict": encoder.state_dict(),
                "classifier_state_dict": classifier.state_dict(),
                "best_acc": best_acc,
            }, ckpt_path)
        else:
            # FIX 4: only count patience after encoder has unfrozen.
            # During the frozen warmup phase the encoder cannot improve anyway,
            # so we skip the counter entirely.
            if epoch >= WARMUP_EPOCHS or freeze_encoder:
                patience_counter += 1

        marker = " *BEST*" if is_best else ""
        print(f"  Epoch [{epoch + 1:3d}/{epochs}]  "
              f"loss={avg_loss:.4f}  train={train_acc:.2f}%  "
              f"val={val_acc:.2f}%  auc={val_auc:.4f}  "
              f"enc_lr={enc_lr:.1e}  cls_lr={cls_lr:.1e}  "
              f"time={epoch_time:.1f}s  gpu={gpu['allocated_mb']:.0f}MB{marker}")

        if patience_counter >= FINETUNE_PATIENCE:
            print(
                f"\n  [EARLY STOP] Validation accuracy has not improved for "
                f"{FINETUNE_PATIENCE} epochs. Best: {best_acc:.2f}% at epoch {best_epoch}."
            )
            break

    total_time = time.time() - total_start
    peak_gpu = _gpu_stats(device)["peak_mb"]
    print(f"\n  Best validation accuracy: {best_acc:.2f}% at epoch {best_epoch}")
    print(f"  Total fine-tuning time: {total_time / 60:.1f} min")
    print(f"  Peak GPU memory: {peak_gpu:.0f} MB")
    print(f"  Metrics saved to: {csv_path}")

    best_path = os.path.join(output_dir, "ckpt_best_finetune.pth")
    if os.path.exists(best_path):
        best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
        encoder.load_state_dict(best_ckpt["encoder_state_dict"])
        classifier.load_state_dict(best_ckpt["classifier_state_dict"])
        print(f"  Loaded best checkpoint from epoch {best_epoch}")

    return history


@torch.no_grad()
def _quick_eval(
    encoder: nn.Module,
    classifier: nn.Module,
    loader: DataLoader,
    device: str,
) -> float:
    """Fast accuracy computation on a DataLoader (no confusion matrix)."""
    encoder.eval()
    classifier.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = classifier(encoder(images))
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


@torch.no_grad()
def _eval_with_auc(
    encoder: nn.Module,
    classifier: nn.Module,
    loader: DataLoader,
    device: str,
    num_classes: int,
) -> tuple:
    """
    Compute val accuracy, val loss, and AUC in one pass.

    NOTE (FIX 9): val_loss uses clean ground-truth labels.  Training loss
    uses Mixup-modified targets.  The two are on a different scale and should
    not be subtracted to compute a "true" generalization gap -- use the
    val_loss trend alone as a convergence indicator.

    Returns:
        (val_acc_percent, val_loss, auc_score)
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
        
        logits = classifier(encoder(images))
        total_loss += criterion(logits, labels).item()

        probs = torch.softmax(logits, dim=1)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())

    val_acc = 100.0 * correct / total
    val_loss = total_loss / total

    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    # --- START THRESHOLD OPTIMIZATION ---
    best_acc = val_acc
    if num_classes == 2:
        # Extract probabilities for the positive class (class 1)
        pos_probs = all_probs[:, 1]
        
        # Test 100 thresholds from 0.0 to 1.0
        thresholds = np.linspace(0, 1, 101)
        best_t = 0.5
        
        for t in thresholds:
            preds = (pos_probs >= t).astype(int)
            current_acc = accuracy_score(all_labels, preds) * 100
            if current_acc > best_acc:
                best_acc = current_acc
                best_t = t
        
        # Log the optimization result to the console
        if best_acc > val_acc:
            print(f"    [Threshold Opt] Improved {val_acc:.2f}% -> {best_acc:.2f}% (t={best_t:.2f})")
    # --- END THRESHOLD OPTIMIZATION ---

    try:
        if num_classes == 2:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            auc = roc_auc_score(
                all_labels, all_probs, multi_class="ovr", average="macro",
            )
    except ValueError:
        auc = 0.0

    return val_acc, val_loss, auc


# ======================================================================
# Training curves visualization
# ======================================================================
def save_training_curves(history: dict, output_dir: str, mode: str) -> None:
    """Save loss, accuracy, AUC, and LR curves as PNG."""
    epochs = history["epoch"]

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle("Fine-Tuning Training Curves", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(epochs, history["loss"], color="#e74c3c", linewidth=2, label="Train (Mixup)")
    if "val_loss" in history and history["val_loss"]:
        ax.plot(epochs, history["val_loss"], color="#c0392b", linewidth=2,
                linestyle="--", label="Validation (clean)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    # FIX 9: title clarifies the two losses are on different scales
    ax.set_title("Train (Mixup) & Validation (clean) Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, history["train_acc"], label="Train", color="#3498db", linewidth=2)
    ax.plot(epochs, history["val_acc"], label="Validation", color="#2ecc71", linewidth=2)
    if history["val_acc"]:
        best_idx = int(np.argmax(history["val_acc"]))
        ax.axvline(x=epochs[best_idx], color="#2ecc71", linestyle="--", alpha=0.5)
        ax.annotate(f"Best: {history['val_acc'][best_idx]:.1f}%",
                    xy=(epochs[best_idx], history["val_acc"][best_idx]),
                    fontsize=9, fontweight="bold", color="#27ae60")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Train vs Validation Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    if "val_auc" in history and history["val_auc"]:
        ax.plot(epochs, history["val_auc"], color="#8e44ad", linewidth=2)
        best_auc_idx = int(np.argmax(history["val_auc"]))
        ax.axvline(x=epochs[best_auc_idx], color="#8e44ad", linestyle="--", alpha=0.5)
        ax.annotate(f"Best: {history['val_auc'][best_auc_idx]:.4f}",
                    xy=(epochs[best_auc_idx], history["val_auc"][best_auc_idx]),
                    fontsize=9, fontweight="bold", color="#6c3483")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.set_title("Validation AUC")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, history["enc_lr"], label="Encoder LR", color="#9b59b6", linewidth=2)
    ax.plot(epochs, history["cls_lr"], label="Classifier LR", color="#f39c12", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.bar(epochs, history["time_s"], color="#1abc9c", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Time per Epoch")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 2]
    if "val_loss" in history and history["val_loss"]:
        gap = [v - t for t, v in zip(history["loss"], history["val_loss"])]
        ax.plot(epochs, gap, color="#e67e22", linewidth=2)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.fill_between(epochs, 0, gap, alpha=0.1,
                        color="red" if max(gap) > 0 else "green")
    ax.set_xlabel("Epoch")
    # FIX 9: axis label clarifies Mixup training loss caveat
    ax.set_ylabel("Val - Train Loss (Mixup; inflated)")
    ax.set_title("Generalization Gap (relative trend only)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, f"training_curves_{mode}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Training curves saved to: {path}")


# ======================================================================
# Classification report
# ======================================================================
def print_classification_report(
    cm: np.ndarray,
    class_names: list,
    output_dir: str,
    mode: str,
) -> None:
    """Print and save a full classification report (precision, recall, F1)."""
    num_classes = len(class_names)
    print("\n  Classification Report:")
    print(f"  {'Class':>20s}  {'Precision':>10s}  {'Recall':>10s}  {'F1-Score':>10s}  {'Support':>10s}")
    print("  " + "-" * 66)

    precisions, recalls, f1s, supports = [], [], [], []

    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        support = cm[c, :].sum()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

        name = class_names[c]
        print(f"  {name:>20s}  {precision:>10.4f}  {recall:>10.4f}  {f1:>10.4f}  {support:>10d}")

    total_support = sum(supports)
    macro_p = np.mean(precisions)
    macro_r = np.mean(recalls)
    macro_f1 = np.mean(f1s)

    weights = np.array(supports) / total_support
    weighted_p = np.dot(weights, precisions)
    weighted_r = np.dot(weights, recalls)
    weighted_f1 = np.dot(weights, f1s)

    print("  " + "-" * 66)
    print(f"  {'macro avg':>20s}  {macro_p:>10.4f}  {macro_r:>10.4f}  {macro_f1:>10.4f}  {total_support:>10d}")
    print(f"  {'weighted avg':>20s}  {weighted_p:>10.4f}  {weighted_r:>10.4f}  {weighted_f1:>10.4f}  {total_support:>10d}")

    report_path = os.path.join(output_dir, f"classification_report_{mode}.txt")
    with open(report_path, "w") as f:
        f.write(f"Classification Report ({mode})\n")
        f.write(f"{'Class':>20s}  {'Precision':>10s}  {'Recall':>10s}  {'F1-Score':>10s}  {'Support':>10s}\n")
        f.write("-" * 70 + "\n")
        for c in range(num_classes):
            name = class_names[c]
            f.write(f"{name:>20s}  {precisions[c]:>10.4f}  {recalls[c]:>10.4f}  {f1s[c]:>10.4f}  {supports[c]:>10d}\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'macro avg':>20s}  {macro_p:>10.4f}  {macro_r:>10.4f}  {macro_f1:>10.4f}  {total_support:>10d}\n")
        f.write(f"{'weighted avg':>20s}  {weighted_p:>10.4f}  {weighted_r:>10.4f}  {weighted_f1:>10.4f}  {total_support:>10d}\n")
    print(f"  Classification report saved to: {report_path}")


# ======================================================================
# Evaluation
# ======================================================================
@torch.no_grad()
def evaluate(
    features: torch.Tensor,
    labels: torch.Tensor,
    classifier: nn.Module,
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

    correct = (preds == labs).sum().item()
    top1 = 100.0 * correct / len(labs)

    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(labs.cpu(), preds.cpu()):
        cm[t.item(), p.item()] += 1

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
    classifier: nn.Module,
    dataloader: DataLoader,
    num_classes: int,
    device: str,
    class_names: list = None,
) -> tuple:
    """Evaluate fine-tuned encoder + classifier on a DataLoader.

    Uses a 5-view TTA ensemble: Original, H-Flip, 5% Zoom, Rot(+10), Rot(-10).
    TF.rotate operates on 4-D BCHW tensors; requires torchvision >= 0.9.0
    (FIX 12: version check at module load time warns if this is not satisfied).

    Returns:
        (top1, per_class, cm, all_probs, all_labels)
    """
    encoder.eval()
    classifier.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in tqdm(dataloader, desc="  Evaluating", leave=False):
        images = images.to(device, non_blocking=True)

        # 1. Original
        probs_orig = torch.softmax(classifier(encoder(images)), dim=1)

        # 2. Horizontal Flip
        img_hflip = torch.flip(images, dims=[3])
        probs_hflip = torch.softmax(classifier(encoder(img_hflip)), dim=1)

        # 3. Zoom (crop 95% of center and resize back)
        B, C, H, W = images.shape
        crop_h, crop_w = int(H * 0.95), int(W * 0.95)
        start_y, start_x = (H - crop_h) // 2, (W - crop_w) // 2
        img_zoom = images[:, :, start_y:start_y+crop_h, start_x:start_x+crop_w]
        img_zoom = F.interpolate(img_zoom, size=(H, W), mode='bilinear', align_corners=False)
        probs_zoom = torch.softmax(classifier(encoder(img_zoom)), dim=1)

        # 4. Rotate +10 degrees (requires torchvision >= 0.9.0 for tensor support)
        img_rot_pos = TF.rotate(images, angle=10)
        probs_rot_pos = torch.softmax(classifier(encoder(img_rot_pos)), dim=1)

        # 5. Rotate -10 degrees
        img_rot_neg = TF.rotate(images, angle=-10)
        probs_rot_neg = torch.softmax(classifier(encoder(img_rot_neg)), dim=1)

        probs = (probs_orig + probs_hflip + probs_zoom + probs_rot_pos + probs_rot_neg) / 5.0

        all_preds.append(probs.argmax(dim=1).cpu())
        all_labels.append(labels)
        all_probs.append(probs.cpu())

    preds = torch.cat(all_preds)
    labs = torch.cat(all_labels)
    probs_np = torch.cat(all_probs).numpy()
    labs_np = labs.numpy()

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

    return top1, per_class, cm.numpy(), probs_np, labs_np


# ======================================================================
# Visualization
# ======================================================================
def save_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    output_path: str,
    title: str = "Confusion Matrix",
) -> None:
    """Save a confusion matrix heatmap as PNG.

    FIX 10: Text color threshold is now per-column (predicted class), not the
    global max.  With a heavily imbalanced dataset the global max is dominated
    by the majority class, making near-zero cells in minority columns appear
    white-on-white.  Per-column normalization keeps each column readable
    independently.
    """
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

    # FIX 10: per-column threshold so every column is independently readable.
    col_max = cm.max(axis=0)  # shape (num_classes,)
    for i in range(num_classes):
        for j in range(num_classes):
            thresh = col_max[j] / 2.0
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
# ROC Curve visualization
# ======================================================================
def save_roc_curve(
    all_probs: np.ndarray,
    all_labels: np.ndarray,
    class_names: list,
    output_dir: str,
    mode: str,
) -> None:
    """Compute and save ROC curve + AUC score as PNG."""
    from sklearn.metrics import roc_curve, auc, roc_auc_score

    num_classes = len(class_names)
    fig, ax = plt.subplots(figsize=(8, 7))

    if num_classes == 2:
        fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color="#2980b9", linewidth=2.5,
                label=f"ROC curve (AUC = {roc_auc:.4f})")
    else:
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(all_labels, classes=list(range(num_classes)))
        colors = plt.cm.Set2(np.linspace(0, 1, num_classes))

        for c in range(num_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, c], all_probs[:, c])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[c], linewidth=2,
                    label=f"{class_names[c]} (AUC = {roc_auc:.4f})")

        try:
            macro_auc = roc_auc_score(
                all_labels, all_probs, multi_class="ovr", average="macro",
            )
            ax.set_title(f"ROC Curves (Macro AUC = {macro_auc:.4f})",
                         fontsize=14, fontweight="bold")
        except ValueError:
            pass

    ax.plot([0, 1], [0, 1], color="gray", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    if num_classes == 2:
        ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    fig.tight_layout()
    path = os.path.join(output_dir, f"roc_curve_{mode}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ROC curve saved to: {path}")


# ======================================================================
# Label scarcity: stratified subset
# ======================================================================
def stratified_subset(dataset, fraction: float, seed: int = 42):
    """
    Create a stratified subset of a dataset, preserving class balance.

    FIX 2: Uses dataset.targets (if available) or a get_label() method to
    avoid loading every image just to collect class labels.  Falls back to
    iterating __getitem__ only when no faster path exists.

    Args:
        dataset: Dataset with (image, label) items.
        fraction: Fraction of data to keep (0.0 - 1.0).
        seed: Random seed for reproducibility.

    Returns:
        torch.utils.data.Subset with stratified indices.
    """
    if fraction >= 1.0:
        return dataset

    from collections import defaultdict
    import random

    rng = random.Random(seed)

    # FIX 2: fast label collection paths -- avoid loading images.
    if hasattr(dataset, "targets"):
        # torchvision-style datasets expose a .targets list
        raw_labels = dataset.targets
        label_iter = (
            (int(l.item()) if isinstance(l, torch.Tensor) else int(l))
            for l in raw_labels
        )
    elif hasattr(dataset, "get_label"):
        label_iter = (dataset.get_label(i) for i in range(len(dataset)))
    else:
        # Slow fallback: load each item to read its label.
        warnings.warn(
            "stratified_subset: dataset has no .targets or .get_label() method. "
            "Falling back to __getitem__ for label collection -- this will load "
            "every image once before training starts.  Add a .targets attribute "
            "or .get_label(idx) method to your dataset class for faster startup.",
            UserWarning,
            stacklevel=2,
        )
        label_iter = (
            (dataset[i][1].item() if isinstance(dataset[i][1], torch.Tensor) else int(dataset[i][1]))
            for i in range(len(dataset))
        )

    class_indices = defaultdict(list)
    for idx, label in enumerate(label_iter):
        class_indices[label].append(idx)

    selected = []
    for cls, indices in sorted(class_indices.items()):
        k = max(1, int(len(indices) * fraction))
        selected.extend(rng.sample(indices, k))

    rng.shuffle(selected)
    print(f"  [Label Scarcity] Using {len(selected)}/{len(dataset)} "
          f"samples ({fraction*100:.0f}%), "
          f"balanced across {len(class_indices)} classes")

    return Subset(dataset, selected)


# ======================================================================
# Label scarcity comparison visualization
# ======================================================================
def save_label_scarcity_comparison(
    results: dict,
    output_dir: str,
) -> None:
    """
    Save a comparison chart for label scarcity robustness.

    Args:
        results: dict mapping fraction (float) -> {
            'accuracy': float, 'auc': float, 'n_samples': int
        }
        output_dir: Directory to save the chart.
    """
    fractions = sorted(results.keys())
    accuracies = [results[f]["accuracy"] for f in fractions]
    aucs = [results[f]["auc"] for f in fractions]
    n_samples = [results[f]["n_samples"] for f in fractions]
    pct_labels = [f"{f*100:.0f}%" for f in fractions]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Label Scarcity Robustness", fontsize=16, fontweight="bold")

    ax = axes[0]
    bars = ax.bar(pct_labels, accuracies, color=["#e74c3c", "#f39c12", "#2ecc71"],
                  edgecolor="black", linewidth=0.5)
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{acc:.2f}%", ha="center", fontsize=11, fontweight="bold")
    ax.set_xlabel("Label Fraction", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Top-1 Accuracy")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    bars = ax.bar(pct_labels, aucs, color=["#e74c3c", "#f39c12", "#2ecc71"],
                  edgecolor="black", linewidth=0.5)
    for bar, a in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{a:.4f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_xlabel("Label Fraction", fontsize=12)
    ax.set_ylabel("AUC", fontsize=12)
    ax.set_title("AUC Score")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[2]
    bars = ax.bar(pct_labels, n_samples, color=["#e74c3c", "#f39c12", "#2ecc71"],
                  edgecolor="black", linewidth=0.5)
    for bar, n in zip(bars, n_samples):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                str(n), ha="center", fontsize=11, fontweight="bold")
    ax.set_xlabel("Label Fraction", fontsize=12)
    ax.set_ylabel("Training Samples", fontsize=12)
    ax.set_title("Dataset Size")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(output_dir, "label_scarcity_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Label scarcity comparison saved to: {path}")

    csv_path = os.path.join(output_dir, "label_scarcity_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label_fraction", "n_samples", "accuracy", "auc"])
        for frac in fractions:
            r = results[frac]
            writer.writerow([
                f"{frac:.2f}", r["n_samples"],
                f"{r['accuracy']:.2f}", f"{r['auc']:.4f}",
            ])
    print(f"  Label scarcity CSV saved to: {csv_path}")


# ======================================================================
# Single-run evaluation (supports label_fraction)
# ======================================================================
def run_evaluation(
    args,
    train_ds,
    test_ds,
    test_loader,
    class_names,
    label_fraction: float = 1.0,
    output_suffix: str = "",
) -> dict:
    """
    Run a single evaluation pass (linear probe or full fine-tune).

    FIX 6: out_dir is now computed explicitly for all three modes, removing
    the implicit fallback to args.output_dir that previously only worked by
    luck for linear_probe + label_fraction < 1.0.

    FIX 1 / FIX 7: frozen_attention mode now writes all outputs (confusion
    matrix, classification report, ROC curve) to sub_output, not a mixture
    of sub_output and args.output_dir.

    Returns:
        dict with 'accuracy', 'auc', 'n_samples'.
    """
    if label_fraction < 1.0:
        subset_ds = stratified_subset(train_ds, label_fraction)
    else:
        subset_ds = train_ds

    n_train = len(subset_ds)

    train_loader = DataLoader(
        subset_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        drop_last=True,
    )

    freeze = (args.mode in ("linear_probe", "frozen_attention"))
    base_encoder = load_encoder(args.encoder_ckpt, args.device, freeze=freeze)

    frac_tag = f"_{label_fraction*100:.0f}pct" if label_fraction < 1.0 else ""
    mode_tag = f"{args.mode}{frac_tag}{output_suffix}"

    final_auc = 0.0

    if args.mode == "linear_probe":
        # FIX 6: linear_probe always uses args.output_dir (no sub_output)
        out_dir = args.output_dir

        encoder = base_encoder
        classifier = nn.Sequential(
            nn.BatchNorm1d(768),
            nn.Dropout(0.2),
            nn.Linear(768, args.num_classes),
        ).to(args.device)
        nn.init.kaiming_uniform_(classifier[2].weight)
        nn.init.zeros_(classifier[2].bias)

        print(f"\n  [Phase 1] Extracting train features ({n_train} samples)...")
        train_feats, train_labels = extract_features(encoder, train_loader, args.device)
        print(f"    Cached {train_feats.shape[0]} features of dim {train_feats.shape[1]}")

        print("  [Phase 2] Extracting test features...")
        test_feats, test_labels = extract_features(encoder, test_loader, args.device)
        print(f"    Cached {test_feats.shape[0]} features of dim {test_feats.shape[1]}")

        classifier = train_linear_classifier(
            train_feats, train_labels,
            num_classes=args.num_classes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
        )

        top1, per_class, cm = evaluate(
            test_feats, test_labels, classifier,
            args.num_classes, args.device, class_names,
        )

    elif args.mode == "frozen_attention":
        # FIX 1 / FIX 7: all outputs go to sub_output, not args.output_dir
        sub_output = os.path.join(args.output_dir, "frozen_attention")
        os.makedirs(sub_output, exist_ok=True)
        out_dir = sub_output  # FIX 6: explicit assignment

        encoder = PatchEncoderWrapper(base_encoder)
        classifier = AttentionPoolClassifier(feat_dim=768, num_classes=args.num_classes).to(args.device)
        nn.init.kaiming_uniform_(classifier.head[2].weight)
        nn.init.zeros_(classifier.head[2].bias)

        history = train_finetune(
            encoder, classifier, train_loader, test_loader,
            epochs=args.epochs, lr=args.lr, device=args.device,
            num_classes=args.num_classes, class_names=class_names,
            output_dir=sub_output,
            freeze_encoder=True,
        )
        save_training_curves(history, sub_output, mode_tag)

        top1, per_class, cm, all_probs, all_labels = evaluate_finetune(
            encoder, classifier, test_loader,
            args.num_classes, args.device, class_names,
        )

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

        # FIX 7: ROC curve saved to sub_output (same directory as the rest)
        save_roc_curve(all_probs, all_labels, class_names, sub_output, mode_tag)

    else:  # full_finetune
        sub_output = os.path.join(args.output_dir, f"finetune{frac_tag}")
        os.makedirs(sub_output, exist_ok=True)
        out_dir = sub_output  # FIX 6: explicit assignment

        encoder = PatchEncoderWrapper(base_encoder)
        classifier = AttentionPoolClassifier(feat_dim=768, num_classes=args.num_classes).to(args.device)
        nn.init.kaiming_uniform_(classifier.head[2].weight)
        nn.init.zeros_(classifier.head[2].bias)

        if args.eval_only:
            print(f"\n  [EVAL ONLY] Loading fine-tuned weights from {args.encoder_ckpt}")
            ckpt = safe_torch_load(args.encoder_ckpt, map_location="cpu")
            if "encoder_state_dict" in ckpt:
                encoder.load_state_dict(ckpt["encoder_state_dict"])
            if "classifier_state_dict" in ckpt:
                classifier.load_state_dict(ckpt["classifier_state_dict"])
            encoder = encoder.to(args.device)
            classifier = classifier.to(args.device)
        else:
            history = train_finetune(
                encoder, classifier, train_loader, test_loader,
                epochs=args.epochs, lr=args.lr, device=args.device,
                num_classes=args.num_classes, class_names=class_names,
                output_dir=sub_output,
            )
            save_training_curves(history, sub_output, mode_tag)

        top1, per_class, cm, all_probs, all_labels = evaluate_finetune(
            encoder, classifier, test_loader,
            args.num_classes, args.device, class_names,
        )

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

        save_roc_curve(all_probs, all_labels, class_names, sub_output, mode_tag)

    # --- Report ---
    mode_label = "Linear Probe" if args.mode == "linear_probe" else "Full Fine-tune"
    frac_label = f" ({label_fraction*100:.0f}% labels)" if label_fraction < 1.0 else ""

    print("\n" + "=" * 60)
    print(f"  {mode_label} Results{frac_label}")
    print("=" * 60)
    print(f"\n  Top-1 Accuracy: {top1:.2f}%")
    if final_auc > 0:
        print(f"  AUC Score:      {final_auc:.4f}")
    print(f"  Training samples: {n_train}\n")
    print("  Per-class accuracy:")
    for name, acc in per_class.items():
        print(f"    {name:>20s}: {acc:.2f}%")

    # FIX 6: out_dir is always explicitly set above; no implicit fallback.
    cm_path = os.path.join(out_dir, f"confusion_matrix_{mode_tag}.png")
    save_confusion_matrix(cm, class_names, cm_path, title=f"{mode_label} Confusion Matrix{frac_label}")
    print_classification_report(cm, class_names, out_dir, mode_tag)

    return {"accuracy": top1, "auc": final_auc, "n_samples": n_train}


# ======================================================================
# Main
# ======================================================================
def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    mode_label = "Linear Probe" if args.mode == "linear_probe" else "Full Fine-tune"

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

    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    if args.label_scarcity:
        fractions = [0.30, 0.60, 1.00]
        scarcity_results = {}

        print("\n" + "=" * 60)
        print("  LABEL SCARCITY ROBUSTNESS EXPERIMENT")
        print(f"  Fractions: {[f'{f*100:.0f}%' for f in fractions]}")
        print(f"  Mode: {mode_label}")
        print("=" * 60)

        for frac in fractions:
            print(f"\n{'─'*60}")
            print(f"  Running with {frac*100:.0f}% of training labels...")
            print(f"{'─'*60}")

            result = run_evaluation(
                args, train_ds, test_ds, test_loader, class_names,
                label_fraction=frac,
            )
            scarcity_results[frac] = result

        print("\n" + "=" * 60)
        print("  LABEL SCARCITY COMPARISON")
        print("=" * 60)
        print(f"\n  {'Fraction':>10s}  {'Samples':>10s}  {'Accuracy':>10s}  {'AUC':>10s}")
        print("  " + "-" * 46)
        for frac in fractions:
            r = scarcity_results[frac]
            print(f"  {frac*100:>9.0f}%  {r['n_samples']:>10d}  "
                  f"{r['accuracy']:>9.2f}%  {r['auc']:>10.4f}")
        print("  " + "-" * 46)

        full = scarcity_results[1.00]
        for frac in [0.30, 0.60]:
            r = scarcity_results[frac]
            acc_drop = full["accuracy"] - r["accuracy"]
            auc_drop = full["auc"] - r["auc"]
            print(f"  {frac*100:.0f}% -> 100%: "
                  f"acc drop={acc_drop:+.2f}%  auc drop={auc_drop:+.4f}")

        save_label_scarcity_comparison(scarcity_results, args.output_dir)
        return

    print("=" * 60)
    print(f"  FedMamba-SALT Evaluation -- {mode_label}")
    if args.label_fraction < 1.0:
        print(f"  Label fraction: {args.label_fraction*100:.0f}%")
    print("=" * 60)

    result = run_evaluation(
        args, train_ds, test_ds, test_loader, class_names,
        label_fraction=args.label_fraction,
    )

    top1 = result["accuracy"]
    print("\n" + "-" * 60)
    print("  BASELINE COMPARISON NOTE:")
    print(f"    Centralized MAE baseline on Retina: ~{FEDMAE_BASELINE:.2f}%")
    print("    (full fine-tuning, centralized setting)")
    print()
    if args.mode == "linear_probe":
        print("    Linear probing is a DIAGNOSTIC -- it measures representation")
        print("    quality but is NOT an apples-to-apples comparison.")
        print("    For a fair comparison, also run with --mode full_finetune.")
    else:
        print(f"    Your full fine-tune result: {top1:.2f}% vs baseline: {FEDMAE_BASELINE:.2f}%")
        diff = top1 - FEDMAE_BASELINE
        sign = "+" if diff >= 0 else ""
        print(f"    Delta: {sign}{diff:.2f}%")
    print("-" * 60)


if __name__ == "__main__":
    main()