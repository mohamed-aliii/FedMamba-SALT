"""
Diagnostic script: Test whether the TEACHER's own frozen features
are linearly separable on the Retina dataset.

If the teacher's features aren't linearly separable, then distilling
those features into the student also produces non-separable features.
The linear probe will be random regardless of how good the distillation is.

This script also checks the ENCODER output statistics to detect
if the projection head is masking encoder collapse.

Run on Colab:
    python diagnostic_teacher_probe.py \
        --data_path /content/drive/MyDrive/Retina \
        --teacher_ckpt data/ckpts/mae_vit_base.pth \
        --student_ckpt outputs/retina_centralized/ckpt_latest.pth
"""

import argparse
import sys
from pathlib import Path

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from augmentations.retina_dataset import RetinaDataset
from augmentations.medical_aug import RETINA_MEAN, RETINA_STD
from models.inception_mamba import InceptionMambaEncoder
from models.vit_teacher import FrozenViTTeacher
from objectives.salt_loss import ProjectionHead
from utils.ckpt_compat import safe_torch_load


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--teacher_ckpt", type=str, required=True)
    p.add_argument("--student_ckpt", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=256)
    return p.parse_args()


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=RETINA_MEAN, std=RETINA_STD),
    ])


@torch.no_grad()
def extract_all(model, loader, device):
    """Extract features and labels from a model."""
    model.eval()
    all_feats, all_labels = [], []
    for images, labels in tqdm(loader, desc="Extracting features"):
        images = images.to(device, non_blocking=True)
        feats = model(images)
        all_feats.append(feats.cpu())
        all_labels.append(labels)
    return torch.cat(all_feats), torch.cat(all_labels)


def train_linear_probe(features, labels, num_classes, epochs=100, lr=1e-3):
    """Train a simple linear classifier and return accuracy."""
    feat_dim = features.shape[1]
    
    # Try multiple classifier heads
    results = {}
    
    # 1. Raw linear
    clf = nn.Linear(feat_dim, num_classes)
    nn.init.kaiming_uniform_(clf.weight)
    nn.init.zeros_(clf.bias)
    opt = Adam(clf.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    for epoch in range(epochs):
        for feats, labs in loader:
            logits = clf(feats)
            loss = criterion(logits, labs)
            opt.zero_grad()
            loss.backward()
            opt.step()
    
    with torch.no_grad():
        preds = clf(features).argmax(dim=1)
        acc = (preds == labels).float().mean().item() * 100
    results["raw_linear"] = acc
    
    # 2. LayerNorm + Linear (what our eval uses)
    clf2 = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, num_classes))
    nn.init.kaiming_uniform_(clf2[1].weight)
    nn.init.zeros_(clf2[1].bias)
    opt2 = Adam(clf2.parameters(), lr=lr, weight_decay=0.0)
    
    for epoch in range(epochs):
        for feats, labs in loader:
            logits = clf2(feats)
            loss = criterion(logits, labs)
            opt2.zero_grad()
            loss.backward()
            opt2.step()
    
    with torch.no_grad():
        preds = clf2(features).argmax(dim=1)
        acc = (preds == labels).float().mean().item() * 100
    results["layernorm_linear"] = acc
    
    # 3. Z-score normalized + Linear
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True).clamp(min=1e-6)
    features_norm = (features - mean) / std
    
    clf3 = nn.Linear(feat_dim, num_classes)
    nn.init.kaiming_uniform_(clf3.weight)
    nn.init.zeros_(clf3.bias)
    opt3 = Adam(clf3.parameters(), lr=lr, weight_decay=0.0)
    dataset_norm = TensorDataset(features_norm, labels)
    loader_norm = DataLoader(dataset_norm, batch_size=256, shuffle=True)
    
    for epoch in range(epochs):
        for feats, labs in loader_norm:
            logits = clf3(feats)
            loss = criterion(logits, labs)
            opt3.zero_grad()
            loss.backward()
            opt3.step()
    
    with torch.no_grad():
        preds = clf3(features_norm).argmax(dim=1)
        acc = (preds == labels).float().mean().item() * 100
    results["zscore_linear"] = acc
    
    return results


def feature_statistics(features, labels, name):
    """Print comprehensive feature statistics."""
    print(f"\n  {'='*55}")
    print(f"  Feature Statistics: {name}")
    print(f"  {'='*55}")
    print(f"    Shape:       {features.shape}")
    print(f"    Global mean: {features.mean().item():.6f}")
    print(f"    Global std:  {features.std().item():.6f}")
    print(f"    Per-dim std: {features.std(dim=0).mean().item():.6f}")
    print(f"    Min:         {features.min().item():.6f}")
    print(f"    Max:         {features.max().item():.6f}")
    print(f"    Norm (avg):  {features.norm(dim=1).mean().item():.4f}")
    
    # Check for collapse: how many dimensions have near-zero variance?
    dim_std = features.std(dim=0)
    dead_dims = (dim_std < 1e-4).sum().item()
    low_dims = (dim_std < 1e-2).sum().item()
    print(f"    Dead dims (std<1e-4):  {dead_dims}/{features.shape[1]}")
    print(f"    Low dims  (std<1e-2):  {low_dims}/{features.shape[1]}")
    
    # Per-class statistics
    unique_labels = labels.unique()
    for lbl in unique_labels:
        mask = labels == lbl
        class_feats = features[mask]
        print(f"    Class {lbl.item()}: n={mask.sum().item()}, "
              f"mean_norm={class_feats.norm(dim=1).mean().item():.4f}, "
              f"mean_val={class_feats.mean().item():.6f}")
    
    # Inter-class cosine similarity
    if len(unique_labels) == 2:
        f0 = features[labels == unique_labels[0]].mean(dim=0)
        f1 = features[labels == unique_labels[1]].mean(dim=0)
        cos = torch.nn.functional.cosine_similarity(f0.unsqueeze(0), f1.unsqueeze(0))
        print(f"    Class-mean cosine sim: {cos.item():.6f}")
        
        # L2 distance between class centers
        dist = (f0 - f1).norm().item()
        print(f"    Class-center L2 dist: {dist:.6f}")
        
        # Centered cosine
        f0c = f0 - features.mean(dim=0)
        f1c = f1 - features.mean(dim=0)
        cos_c = torch.nn.functional.cosine_similarity(f0c.unsqueeze(0), f1c.unsqueeze(0))
        print(f"    Centered cosine sim:  {cos_c.item():.6f}")


def main():
    args = parse_args()
    device = args.device
    transform = get_eval_transform()
    
    # Load train dataset
    train_ds = RetinaDataset(
        data_path=args.data_path, phase="train",
        split_type="central", split_csv="train.csv",
        transform=transform,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    
    print(f"\n{'='*60}")
    print(f"  DIAGNOSTIC: Feature Quality Analysis")
    print(f"{'='*60}")
    
    # =====================================================
    # TEST 1: Teacher's own linear separability
    # =====================================================
    print(f"\n\n{'#'*60}")
    print(f"  TEST 1: Teacher Feature Linear Separability")
    print(f"{'#'*60}")
    
    teacher = FrozenViTTeacher(ckpt_path=args.teacher_ckpt).to(device)
    t_feats, t_labels = extract_all(teacher, train_loader, device)
    feature_statistics(t_feats, t_labels, "Teacher (frozen MAE ViT-B/16)")
    
    print("\n  Training linear probes on TEACHER features (100 epochs)...")
    t_results = train_linear_probe(t_feats, t_labels, num_classes=2)
    for method, acc in t_results.items():
        tag = "PASS" if acc > 60 else "FAIL"
        print(f"    [{tag}] {method}: {acc:.2f}%")
    
    del teacher
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # =====================================================
    # TEST 2: Student encoder output statistics
    # =====================================================
    if args.student_ckpt:
        print(f"\n\n{'#'*60}")
        print(f"  TEST 2: Student Encoder Feature Analysis")
        print(f"{'#'*60}")
        
        encoder = InceptionMambaEncoder(
            patch_size=16, embed_dim=256, depth=4, out_dim=768,
        )
        ckpt = safe_torch_load(args.student_ckpt, map_location="cpu")
        encoder.load_state_dict(ckpt["student_state_dict"])
        encoder = encoder.to(device)
        encoder.eval()
        
        for p in encoder.parameters():
            p.requires_grad = False
        
        s_feats, s_labels = extract_all(encoder, train_loader, device)
        feature_statistics(s_feats, s_labels, "Student Encoder (before proj head)")
        
        print("\n  Training linear probes on STUDENT ENCODER features (100 epochs)...")
        s_results = train_linear_probe(s_feats, s_labels, num_classes=2)
        for method, acc in s_results.items():
            tag = "PASS" if acc > 60 else "FAIL"
            print(f"    [{tag}] {method}: {acc:.2f}%")
        
        # =====================================================
        # TEST 3: Student + Projection Head output
        # =====================================================
        print(f"\n\n{'#'*60}")
        print(f"  TEST 3: Projection Head Output Analysis")
        print(f"{'#'*60}")
        
        if "projector_state_dict" in ckpt:
            projector = ProjectionHead(in_dim=768, hidden_dim=2048, out_dim=768)
            projector.load_state_dict(ckpt["projector_state_dict"])
            projector = projector.to(device)
            projector.eval()
            
            with torch.no_grad():
                proj_feats = projector(s_feats.to(device)).cpu()
            
            feature_statistics(proj_feats, s_labels, "Student + Projection Head")
            
            print("\n  Training linear probes on PROJECTED features (100 epochs)...")
            p_results = train_linear_probe(proj_feats, s_labels, num_classes=2)
            for method, acc in p_results.items():
                tag = "PASS" if acc > 60 else "FAIL"
                print(f"    [{tag}] {method}: {acc:.2f}%")
        else:
            print("  [SKIP] No projector_state_dict in checkpoint.")
        
        # =====================================================
        # TEST 4: Cosine similarity between student and teacher
        # =====================================================
        print(f"\n\n{'#'*60}")
        print(f"  TEST 4: Student-Teacher Alignment Quality")
        print(f"{'#'*60}")
        
        # Re-extract teacher features (same ordering)
        teacher2 = FrozenViTTeacher(ckpt_path=args.teacher_ckpt).to(device)
        t_feats2, _ = extract_all(teacher2, train_loader, device)
        
        # Compare with projection output if available
        if "projector_state_dict" in ckpt:
            compare_feats = proj_feats
            compare_name = "projected"
        else:
            compare_feats = s_feats
            compare_name = "encoder"
        
        cos_sim = torch.nn.functional.cosine_similarity(
            compare_feats, t_feats2, dim=1
        )
        l2_dist = (compare_feats - t_feats2).norm(dim=1)
        
        print(f"  Student ({compare_name}) vs Teacher:")
        print(f"    Cosine similarity: {cos_sim.mean().item():.6f} ± {cos_sim.std().item():.6f}")
        print(f"    L2 distance:       {l2_dist.mean().item():.6f} ± {l2_dist.std().item():.6f}")
        
        del teacher2
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # =====================================================
    # SUMMARY
    # =====================================================
    print(f"\n\n{'='*60}")
    print(f"  DIAGNOSTIC SUMMARY")
    print(f"{'='*60}")
    print(f"  Teacher linear probe (best): {max(t_results.values()):.2f}%")
    if args.student_ckpt:
        print(f"  Student linear probe (best): {max(s_results.values()):.2f}%")
        if "projector_state_dict" in ckpt:
            print(f"  Projected linear probe (best): {max(p_results.values()):.2f}%")
    
    if max(t_results.values()) < 60:
        print("\n  *** CRITICAL: Teacher features are NOT linearly separable! ***")
        print("  This means distillation CANNOT produce linearly separable features.")
        print("  The linear probe will always be ~50% regardless of distillation quality.")
        print("  The 81.93% baseline was achieved by FINE-TUNING the full ViT,")
        print("  not by linear probing frozen features.")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
