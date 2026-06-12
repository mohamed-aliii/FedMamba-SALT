# --------------------------------------------------------
# PEFT Fine-tuning for Fed-MAE with LoRA + FedNMC
#
# Three modes:
#   full_ft     — Full fine-tuning (paper baseline reproduction)
#   lora_naive  — LoRA + Linear head + FedAvg + CE (expected to collapse)
#   lora_fednmc — LoRA + Prototype Cosine classifier + EMA (proposed)
#
# Built on top of the SSL-FL codebase, reusing its data loading,
# federated loop, and evaluation infrastructure.
# --------------------------------------------------------

import argparse
import csv
import datetime
import json
import numpy as np
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

# --- timm 0.3.2 compatibility shim for PyTorch 2.x ---
import collections.abc, math, types
if "torch._six" not in sys.modules:
    _mock = types.ModuleType("torch._six")
    _mock.container_abcs = collections.abc
    _mock.inf = math.inf
    sys.modules["torch._six"] = _mock

import timm
from timm.utils import accuracy
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models.layers import trunc_normal_

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import fed_mae.models_vit as models_vit
from fed_mae.peft_lora import inject_lora, freeze_non_lora, get_lora_state_dict, get_trainable_state_dict
import util.misc as misc
import util.lr_sched as lr_sched
from util.pos_embed import interpolate_pos_embed
from util.data_utils import DatasetFLFinetune, create_dataset_and_evalmetrix
from util.start_config import print_options


# ================================================================
# Prototype Cosine Classifier (for FedNMC mode)
# ================================================================
class PrototypeCosineClassifier(nn.Module):
    """
    Cosine similarity classifier with learnable temperature and
    EMA-aggregated class prototypes.
    """
    def __init__(self, embed_dim: int, num_classes: int, tau: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        # Learnable temperature (log-scale for stability)
        self.log_tau = nn.Parameter(torch.tensor(np.log(tau)))
        # Prototypes — initialized randomly, updated via EMA
        self.prototypes = nn.Parameter(torch.randn(num_classes, embed_dim))
        nn.init.kaiming_uniform_(self.prototypes)
        self.prototypes.requires_grad = False  # Updated via EMA, not gradient

    @property
    def tau(self):
        return self.log_tau.exp().clamp(min=0.01, max=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, D) normalized features -> (B, C) logits."""
        x_norm = F.normalize(x, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)
        logits = torch.mm(x_norm, p_norm.T) / self.tau
        return logits

    @torch.no_grad()
    def update_prototypes_ema(self, features: torch.Tensor, labels: torch.Tensor,
                              momentum: float = 0.9):
        """Update prototypes with EMA from batch features."""
        features = F.normalize(features.detach(), dim=-1)
        for c in range(self.num_classes):
            mask = labels == c
            if mask.sum() > 0:
                class_mean = features[mask].mean(dim=0)
                self.prototypes.data[c] = (
                    momentum * self.prototypes.data[c] +
                    (1 - momentum) * class_mean
                )


# ================================================================
# Per-class metrics tracking
# ================================================================
class MetricsTracker:
    """Tracks per-round metrics including per-class accuracy, confusion matrix."""

    def __init__(self, output_dir: str, num_classes: int, mode: str):
        self.output_dir = output_dir
        self.num_classes = num_classes
        self.mode = mode
        os.makedirs(output_dir, exist_ok=True)

        self.csv_path = os.path.join(output_dir, "metrics.csv")
        header = ["round", "train_loss", "test_acc", "test_loss", "max_acc", "lr"]
        for c in range(num_classes):
            header.extend([f"class{c}_precision", f"class{c}_recall", f"class{c}_f1"])
        header.append("trainable_params")

        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(header)

        self.history = {"rounds": [], "train_loss": [], "test_acc": [],
                        "per_class_recall": [[] for _ in range(num_classes)]}

    def log(self, round_num, train_loss, test_stats, max_acc, lr, n_params,
            per_class_metrics=None):
        row = [round_num, f"{train_loss:.4f}",
               f"{test_stats.get('acc1', 0):.2f}",
               f"{test_stats.get('loss', 0):.4f}",
               f"{max_acc:.2f}", f"{lr:.2e}"]

        if per_class_metrics is not None:
            for c in range(self.num_classes):
                p = per_class_metrics.get(c, {})
                row.extend([
                    f"{p.get('precision', 0):.4f}",
                    f"{p.get('recall', 0):.4f}",
                    f"{p.get('f1', 0):.4f}",
                ])
        else:
            row.extend([""] * (self.num_classes * 3))

        row.append(n_params)

        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        self.history["rounds"].append(round_num)
        self.history["train_loss"].append(train_loss)
        self.history["test_acc"].append(test_stats.get('acc1', 0))
        if per_class_metrics:
            for c in range(self.num_classes):
                self.history["per_class_recall"][c].append(
                    per_class_metrics.get(c, {}).get('recall', 0))

    def save_plots(self):
        """Generate and save publication-quality plots."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("[MetricsTracker] matplotlib not available, skipping plots")
            return

        rounds = self.history["rounds"]
        if len(rounds) < 2:
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Plot 1: Test accuracy over rounds
        axes[0].plot(rounds, self.history["test_acc"], 'b-', linewidth=2)
        axes[0].set_xlabel("Communication Round", fontsize=12)
        axes[0].set_ylabel("Test Accuracy (%)", fontsize=12)
        axes[0].set_title(f"{self.mode}: Test Accuracy", fontsize=14)
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Training loss
        axes[1].plot(rounds, self.history["train_loss"], 'r-', linewidth=2)
        axes[1].set_xlabel("Communication Round", fontsize=12)
        axes[1].set_ylabel("Training Loss", fontsize=12)
        axes[1].set_title(f"{self.mode}: Training Loss", fontsize=14)
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Per-class recall
        colors = plt.cm.Set1(np.linspace(0, 1, self.num_classes))
        for c in range(self.num_classes):
            if self.history["per_class_recall"][c]:
                axes[2].plot(rounds[:len(self.history["per_class_recall"][c])],
                            self.history["per_class_recall"][c],
                            color=colors[c], linewidth=2, label=f"Class {c}")
        axes[2].set_xlabel("Communication Round", fontsize=12)
        axes[2].set_ylabel("Recall", fontsize=12)
        axes[2].set_title(f"{self.mode}: Per-Class Recall", fontsize=14)
        axes[2].legend(fontsize=10)
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "training_curves.png"), dpi=150)
        plt.close()

        # Separate per-class recall plot (larger, for paper)
        fig, ax = plt.subplots(figsize=(8, 6))
        for c in range(self.num_classes):
            if self.history["per_class_recall"][c]:
                ax.plot(rounds[:len(self.history["per_class_recall"][c])],
                        self.history["per_class_recall"][c],
                        color=colors[c], linewidth=2.5, label=f"Class {c}")
        ax.set_xlabel("Communication Round", fontsize=14)
        ax.set_ylabel("Recall", fontsize=14)
        ax.set_title(f"{self.mode}: Per-Class Recall Over Rounds", fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "per_class_recall.png"), dpi=200)
        plt.close()


# ================================================================
# Enhanced evaluation with per-class metrics
# ================================================================
@torch.no_grad()
def evaluate_with_per_class(model, data_loader, device, num_classes,
                            classifier=None):
    """Evaluate model and return per-class precision/recall/F1."""
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    criterion = nn.CrossEntropyLoss()

    for batch in data_loader:
        images, target = batch[0].to(device), batch[-1].to(device)

        with torch.cuda.amp.autocast():
            if classifier is not None:
                features = model.forward_features(images)
                output = classifier(features)
            else:
                output = model(images)

            loss = criterion(output, target)

        _, preds = output.max(1)
        all_preds.append(preds.cpu())
        all_targets.append(target.cpu())
        total_loss += loss.item() * images.size(0)
        total_correct += (preds == target).sum().item()
        total_samples += images.size(0)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    acc = 100.0 * total_correct / total_samples
    avg_loss = total_loss / total_samples

    # Per-class metrics
    per_class = {}
    for c in range(num_classes):
        tp = ((all_preds == c) & (all_targets == c)).sum().item()
        fp = ((all_preds == c) & (all_targets != c)).sum().item()
        fn = ((all_preds != c) & (all_targets == c)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[c] = {"precision": precision, "recall": recall, "f1": f1,
                        "tp": tp, "fp": fp, "fn": fn}

    # Confusion matrix
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(all_targets, all_preds):
        confusion[t.long(), p.long()] += 1

    return {"acc1": acc, "loss": avg_loss}, per_class, confusion


# ================================================================
# FedNMC aggregation: average LoRA params + EMA-aggregate prototypes
# ================================================================
def fednmc_aggregate(model_avg, model_all, classifier_all,
                     args, proto_momentum=0.7):
    """
    Aggregate client models:
    - LoRA + head params: weighted FedAvg
    - Prototypes: weighted average across clients (one-shot, not EMA here)
    """
    device = 'cpu'
    model_avg.to(device)

    # --- Aggregate trainable model params (LoRA + head + fc_norm) ---
    avg_sd = model_avg.state_dict()
    trainable_keys = [k for k in avg_sd.keys()
                      if any(t in k for t in ("lora_", "head.", "fc_norm."))]

    for key in trainable_keys:
        weighted_sum = None
        for client in args.proxy_clients:
            w = torch.tensor(args.clients_weightes[client], dtype=torch.float32)
            client_param = model_all[client].state_dict()[key].cpu().float()
            if weighted_sum is None:
                weighted_sum = client_param * w
            else:
                weighted_sum += client_param * w
        avg_sd[key] = weighted_sum

    model_avg.load_state_dict(avg_sd)

    # Distribute averaged params back to clients
    for client in args.proxy_clients:
        client_sd = model_all[client].state_dict()
        for key in trainable_keys:
            client_sd[key].copy_(avg_sd[key].to(client_sd[key].device))

    # --- Aggregate prototypes via weighted average ---
    if classifier_all is not None:
        avg_proto = None
        for client in args.proxy_clients:
            w = args.clients_weightes[client]
            client_proto = classifier_all[client].prototypes.data.cpu().float()
            if avg_proto is None:
                avg_proto = client_proto * w
            else:
                avg_proto += client_proto * w

        # Distribute aggregated prototypes back
        for client in args.proxy_clients:
            classifier_all[client].prototypes.data.copy_(
                avg_proto.to(classifier_all[client].prototypes.device))


# ================================================================
# Standard FedAvg (for full_ft and lora_naive)
# ================================================================
def fedavg_aggregate(model_avg, model_all, args):
    """Standard FedAvg: weighted average of all model parameters."""
    model_avg.cpu()
    params = dict(model_avg.named_parameters())

    for name, param in params.items():
        weighted_sum = None
        for client in args.proxy_clients:
            w = torch.tensor(args.clients_weightes[client], dtype=torch.float32)
            client_param = dict(model_all[client].named_parameters())[name].data.cpu() * w
            if weighted_sum is None:
                weighted_sum = client_param
            else:
                weighted_sum += client_param
        param.data.copy_(weighted_sum)

    # Distribute back
    for client in args.proxy_clients:
        client_params = dict(model_all[client].named_parameters())
        for name, param in params.items():
            client_params[name].data.copy_(param.data.to(client_params[name].device))


# ================================================================
# Training engine for one client, one round
# ================================================================
def train_one_round_client(model, data_loader, optimizer, device, epoch,
                           loss_scaler, criterion, mixup_fn, classifier,
                           args, proxy_single_client):
    """Train one client for E_epoch local epochs."""
    model.train()
    if classifier is not None:
        classifier.train()

    total_loss = 0.0
    n_batches = 0

    for inner_epoch in range(args.E_epoch):
        for data_iter_step, (samples, targets) in enumerate(data_loader):
            args.global_step_per_client[proxy_single_client] += 1

            # LR schedule
            if data_iter_step % args.accum_iter == 0:
                lr_sched.adjust_learning_rate(
                    optimizer, data_iter_step / len(data_loader) + epoch, args)

            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if mixup_fn is not None and args.peft_mode == 'full_ft':
                samples, targets = mixup_fn(samples, targets)

            with torch.cuda.amp.autocast():
                if classifier is not None:
                    features = model.forward_features(samples)
                    output = classifier(features)
                else:
                    output = model(samples)

                loss = criterion(output, targets)

            loss_value = loss.item()
            if not np.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping")
                sys.exit(1)

            loss /= args.accum_iter
            loss_scaler(loss, optimizer, clip_grad=args.clip_grad,
                        parameters=[p for p in model.parameters() if p.requires_grad] +
                                   ([p for p in classifier.parameters() if p.requires_grad]
                                    if classifier is not None else []),
                        create_graph=False,
                        update_grad=(data_iter_step + 1) % args.accum_iter == 0)

            if (data_iter_step + 1) % args.accum_iter == 0:
                optimizer.zero_grad()

            torch.cuda.synchronize()

            # EMA prototype update (FedNMC only)
            if classifier is not None and isinstance(classifier, PrototypeCosineClassifier):
                with torch.no_grad():
                    feats = model.forward_features(samples)
                    # Need integer targets for EMA update
                    if targets.dim() == 1:
                        classifier.update_prototypes_ema(
                            feats, targets, momentum=args.proto_momentum)

            total_loss += loss_value
            n_batches += 1

    avg_loss = total_loss / max(1, n_batches)
    return avg_loss


# ================================================================
# CLI
# ================================================================
def get_args():
    parser = argparse.ArgumentParser('PEFT Fed-MAE Fine-tuning', add_help=False)

    # PEFT mode
    parser.add_argument('--peft_mode', default='lora_fednmc', type=str,
                        choices=['full_ft', 'lora_naive', 'lora_fednmc'],
                        help='Fine-tuning mode')

    # LoRA config
    parser.add_argument('--lora_rank', default=16, type=int, help='LoRA rank')
    parser.add_argument('--lora_alpha', default=None, type=float,
                        help='LoRA alpha (default: 2*rank)')
    parser.add_argument('--lora_targets', default='qkv_proj', type=str,
                        choices=['qkv', 'qkv_proj', 'all'],
                        help='Which attention modules to adapt with LoRA')

    # FedNMC config
    parser.add_argument('--proto_tau', default=0.1, type=float,
                        help='Temperature for cosine classifier')
    parser.add_argument('--proto_momentum', default=0.9, type=float,
                        help='EMA momentum for prototype updates')

    # Standard fine-tuning args (matching SSL-FL interface)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=10, type=int)

    # Model
    parser.add_argument('--model_name', default='mae', type=str)
    parser.add_argument('--model', default='vit_base_patch16', type=str)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--drop_path', type=float, default=0.1)

    # Optimizer
    parser.add_argument('--clip_grad', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--blr', type=float, default=1e-3,
                        help='base lr: absolute_lr = base_lr * batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=5)

    # Augmentation
    parser.add_argument('--color_jitter', type=float, default=None)
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1')
    parser.add_argument('--smoothing', type=float, default=0.1)
    parser.add_argument('--reprob', type=float, default=0.25)
    parser.add_argument('--remode', type=str, default='pixel')
    parser.add_argument('--recount', type=int, default=1)
    parser.add_argument('--resplit', action='store_true', default=False)

    # Mixup (only used in full_ft mode)
    parser.add_argument('--mixup', type=float, default=0)
    parser.add_argument('--cutmix', type=float, default=0)
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None)
    parser.add_argument('--mixup_prob', type=float, default=1.0)
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5)
    parser.add_argument('--mixup_mode', type=str, default='batch')

    # Checkpoint
    parser.add_argument('--finetune', default='', help='Pretrained checkpoint path')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool')

    # Data
    parser.add_argument('--data_set', default='Retina', type=str)
    parser.add_argument('--data_path', default='', type=str)
    parser.add_argument('--nb_classes', default=2, type=int)
    parser.add_argument('--output_dir', default='', help='Output directory')
    parser.add_argument('--class_weights', type=float, nargs='+', default=None,
                        help='Class weights for CrossEntropyLoss (e.g. 1.0 2.0 10.0)')
    parser.add_argument('--log_dir', default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.set_defaults(pin_mem=True)

    # Distributed (kept for interface compat, single-GPU Colab)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--sync_bn', default=False, action='store_true')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')

    # FL
    parser.add_argument('--n_clients', default=5, type=int)
    parser.add_argument('--E_epoch', default=1, type=int, help='Local epochs per round')
    parser.add_argument('--max_communication_rounds', default=100, type=int)
    parser.add_argument('--num_local_clients', default=-1, type=int)
    parser.add_argument('--split_type', type=str, default='central')

    return parser.parse_args()


# ================================================================
# Main
# ================================================================
def main():
    args = get_args()
    args.distributed = False  # Single-GPU Colab

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    # Set default lora_alpha
    if args.lora_alpha is None:
        args.lora_alpha = float(args.lora_rank * 2)

    # Compute absolute LR
    total_batch_size = args.batch_size * args.accum_iter
    if args.lr is None:
        args.lr = args.blr * total_batch_size / 256
    print(f"Absolute LR = {args.lr:.6f}")

    # Output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Save args
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # ---- Build model ----
    print(f"\n{'='*60}")
    print(f"  PEFT Mode: {args.peft_mode}")
    print(f"{'='*60}")

    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )

    # Load pretrained checkpoint
    if args.finetune:
        # ---- Inline pandas-compat unpickler to avoid missing 'utils' dependency ----
        import pickle
        from typing import Any
        
        def _make_safe_new_block():
            from pandas.core.internals.blocks import new_block as _real_new_block
            from pandas._libs.internals import BlockPlacement
            def _safe_new_block(*b_args, **b_kwargs):
                b_args = list(b_args)
                for i, a in enumerate(b_args):
                    if isinstance(a, slice): b_args[i] = BlockPlacement(a)
                for k, v in b_kwargs.items():
                    if isinstance(v, slice): b_kwargs[k] = BlockPlacement(v)
                return _real_new_block(*b_args, **b_kwargs)
            return _safe_new_block

        class _CompatUnpickler(pickle.Unpickler):
            def find_class(self, module: str, name: str) -> Any:
                if module == "pandas.core.internals.blocks" and name == "new_block":
                    return _make_safe_new_block()
                return super().find_class(module, name)

        class _CompatPickleModule:
            Unpickler = _CompatUnpickler
            def __getattr__(self, name): return getattr(pickle, name)

        try:
            checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=False, pickle_module=_CompatPickleModule())
            print("[safe_torch_load] Successfully used pandas-compat unpickler inline.")
        except Exception as e:
            print(f"Warning: safe load failed ({e}), falling back to standard load")
            checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=False)
        # -----------------------------------------------------------------------------
        checkpoint_model = checkpoint['model']

        # Remove head keys if shape mismatch
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                del checkpoint_model[k]

        interpolate_pos_embed(model, checkpoint_model)
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(f"Loaded checkpoint: {args.finetune}")
        print(f"  Missing: {msg.missing_keys}")
        print(f"  Unexpected: {msg.unexpected_keys}")

    # ---- Apply PEFT mode ----
    classifier_proto = None  # Only used in fednmc mode

    if args.peft_mode == 'full_ft':
        # Full fine-tuning — initialize head
        trunc_normal_(model.head.weight, std=2e-5)
        print(f"Mode: Full fine-tuning (all {sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")

    elif args.peft_mode == 'lora_naive':
        # LoRA + linear head + CE
        trunc_normal_(model.head.weight, std=2e-5)
        lora_info = inject_lora(model, rank=args.lora_rank, alpha=args.lora_alpha,
                                target_modules=args.lora_targets)
        freeze_non_lora(model, freeze_head=False)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Mode: LoRA Naive — rank={args.lora_rank}, targets={args.lora_targets}")
        print(f"  LoRA layers injected: {lora_info['injected_layers']}")
        print(f"  Trainable: {trainable/1e6:.2f}M / {total/1e6:.1f}M ({100*trainable/total:.2f}%)")

    elif args.peft_mode == 'lora_fednmc':
        # LoRA + prototype cosine classifier
        lora_info = inject_lora(model, rank=args.lora_rank, alpha=args.lora_alpha,
                                target_modules=args.lora_targets)
        freeze_non_lora(model, freeze_head=True)  # Freeze the linear head (we use prototypes)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Mode: LoRA FedNMC — rank={args.lora_rank}, targets={args.lora_targets}")
        print(f"  LoRA layers injected: {lora_info['injected_layers']}")
        print(f"  Trainable encoder: {trainable/1e6:.2f}M / {total/1e6:.1f}M ({100*trainable/total:.2f}%)")
        print(f"  Prototype tau={args.proto_tau}, momentum={args.proto_momentum}")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ---- Prepare data ----
    # Initialize tracking dicts required by create_dataset_and_evalmetrix
    args.best_acc = {}
    args.current_acc = {}
    args.current_test_acc = {}
    args.best_mlm_acc = {}
    args.t_total = {}
    create_dataset_and_evalmetrix(args, mode='finetune')

    dataset_val = DatasetFLFinetune(args=args, phase='test')
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=torch.utils.data.SequentialSampler(dataset_val),
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False)

    # ---- Per-client setup ----
    model_all = {}
    optimizer_all = {}
    criterion_all = {}
    loss_scaler_all = {}
    mixup_fn_all = {}
    classifier_all = {} if args.peft_mode == 'lora_fednmc' else None

    # Client selection
    if args.num_local_clients == -1:
        args.proxy_clients = args.dis_cvs_files
        args.num_local_clients = len(args.dis_cvs_files)
    else:
        args.proxy_clients = ['train_' + str(i) for i in range(args.num_local_clients)]

    args.clients_weightes = {}
    args.global_step_per_client = {name: 0 for name in args.proxy_clients}

    for proxy_client in args.proxy_clients:
        num_training_steps = args.clients_with_len[proxy_client] // total_batch_size
        args.t_total[proxy_client] = num_training_steps * args.E_epoch * args.max_communication_rounds

        # Clone model per client
        model_all[proxy_client] = deepcopy(model).to(device)

        # Build optimizer based on mode
        if args.peft_mode == 'full_ft':
            from util.lr_decay import param_groups_lrd
            param_groups = param_groups_lrd(model_all[proxy_client],
                                            args.weight_decay,
                                            no_weight_decay_list=model_all[proxy_client].no_weight_decay(),
                                            layer_decay=args.layer_decay)
            optimizer_all[proxy_client] = torch.optim.AdamW(param_groups, lr=args.lr)
        else:
            # LoRA modes: only optimize trainable params
            trainable_params = [p for p in model_all[proxy_client].parameters() if p.requires_grad]
            optimizer_params = [{"params": trainable_params, "weight_decay": args.weight_decay}]

            if args.peft_mode == 'lora_fednmc':
                classifier_all[proxy_client] = PrototypeCosineClassifier(
                    embed_dim=768, num_classes=args.nb_classes, tau=args.proto_tau
                ).to(device)
                # Add log_tau to optimizer
                optimizer_params.append({
                    "params": [classifier_all[proxy_client].log_tau],
                    "weight_decay": 0.0
                })

            optimizer_all[proxy_client] = torch.optim.AdamW(optimizer_params, lr=args.lr)

        # Build criterion
        if args.peft_mode == 'full_ft':
            mixup_fn = None
            mixup_active = args.mixup > 0 or args.cutmix > 0 or args.cutmix_minmax is not None
            if mixup_active:
                mixup_fn = Mixup(
                    mixup_alpha=args.mixup, cutmix_alpha=args.cutmix,
                    cutmix_minmax=args.cutmix_minmax, prob=args.mixup_prob,
                    switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                    label_smoothing=args.smoothing, num_classes=args.nb_classes)
            mixup_fn_all[proxy_client] = mixup_fn

            if mixup_fn is not None:
                criterion_all[proxy_client] = SoftTargetCrossEntropy()
            elif args.smoothing > 0.:
                criterion_all[proxy_client] = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
            elif args.class_weights is not None:
                weights = torch.tensor(args.class_weights, dtype=torch.float32).to(device)
                criterion_all[proxy_client] = nn.CrossEntropyLoss(weight=weights)
            else:
                criterion_all[proxy_client] = nn.CrossEntropyLoss()
        else:
            mixup_fn_all[proxy_client] = None
            criterion_all[proxy_client] = nn.CrossEntropyLoss()

        loss_scaler_all[proxy_client] = misc.NativeScalerWithGradNormCount()

    model_avg = deepcopy(model).cpu()

    # Metrics tracker
    metrics = MetricsTracker(args.output_dir, args.nb_classes, args.peft_mode)

    # ---- Federated training loop ----
    print(f"\n{'='*60}")
    print(f"  Starting federated fine-tuning: {args.peft_mode}")
    print(f"  Clients: {len(args.proxy_clients)}, Rounds: {args.max_communication_rounds}")
    print(f"  Trainable params: {n_trainable:,}")
    print(f"{'='*60}\n")

    max_accuracy = 0.0
    start_time = time.time()
    tot_clients = args.dis_cvs_files

    for epoch in range(args.max_communication_rounds):
        epoch_start = time.time()

        # Select clients
        if args.num_local_clients == len(tot_clients):
            cur_selected_clients = args.proxy_clients
        else:
            cur_selected_clients = np.random.choice(
                tot_clients, args.num_local_clients, replace=False).tolist()

        # Compute client weights
        cur_tot_len = sum(args.clients_with_len[c] for c in cur_selected_clients)

        avg_train_loss = 0.0
        for cur_client, proxy_client in zip(cur_selected_clients, args.proxy_clients):
            args.single_client = cur_client
            args.clients_weightes[proxy_client] = args.clients_with_len[cur_client] / cur_tot_len

            # Build per-client dataloader
            dataset_train = DatasetFLFinetune(args=args, phase='train')
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            data_loader_train = torch.utils.data.DataLoader(
                dataset_train, sampler=sampler_train,
                batch_size=args.batch_size, num_workers=args.num_workers,
                pin_memory=args.pin_mem, drop_last=True)

            # Local training
            client_classifier = classifier_all[proxy_client] if classifier_all else None
            client_loss = train_one_round_client(
                model_all[proxy_client], data_loader_train,
                optimizer_all[proxy_client], device, epoch,
                loss_scaler_all[proxy_client],
                criterion_all[proxy_client],
                mixup_fn_all[proxy_client],
                client_classifier, args, proxy_client)

            avg_train_loss += client_loss * args.clients_weightes[proxy_client]

        # ---- Aggregation ----
        if args.peft_mode == 'lora_fednmc':
            fednmc_aggregate(model_avg, model_all, classifier_all, args)
        else:
            fedavg_aggregate(model_avg, model_all, args)

        # ---- Evaluation ----
        model_avg.to(device)
        eval_classifier = None
        if args.peft_mode == 'lora_fednmc':
            # Use first client's classifier (all synced after aggregation)
            eval_classifier = classifier_all[args.proxy_clients[0]]

        test_stats, per_class, confusion = evaluate_with_per_class(
            model_avg, data_loader_val, device, args.nb_classes, eval_classifier)
        model_avg.to('cpu')

        # Update best
        if test_stats['acc1'] > max_accuracy:
            max_accuracy = test_stats['acc1']
            # Save best checkpoint
            if args.output_dir:
                save_dict = {
                    'model': model_avg.state_dict(),
                    'args': vars(args),
                    'epoch': epoch,
                    'accuracy': max_accuracy,
                }
                if args.peft_mode == 'lora_fednmc' and eval_classifier is not None:
                    save_dict['classifier'] = eval_classifier.state_dict()
                torch.save(save_dict, os.path.join(args.output_dir, 'checkpoint-best.pth'))

        # Get current LR
        cur_lr = optimizer_all[args.proxy_clients[0]].param_groups[0]['lr']

        # Log metrics
        metrics.log(epoch, avg_train_loss, test_stats, max_accuracy, cur_lr,
                    n_trainable, per_class)

        # Print
        epoch_time = time.time() - epoch_start
        pcr_str = ', '.join(f"{per_class[c]['recall']:.2f}" for c in range(args.nb_classes))
        print(f"Round {epoch:3d}/{args.max_communication_rounds} | "
              f"loss={avg_train_loss:.4f} | acc={test_stats['acc1']:.1f}% | "
              f"best={max_accuracy:.1f}% | "
              f"per_class=[{pcr_str}] | "
              f"lr={cur_lr:.2e} | {epoch_time:.0f}s")

        # Save periodic checkpoint
        if args.output_dir and (epoch + 1) % args.save_ckpt_freq == 0:
            save_dict = {
                'model': model_avg.state_dict(),
                'args': vars(args),
                'epoch': epoch,
                'accuracy': test_stats['acc1'],
            }
            if args.peft_mode == 'lora_fednmc' and eval_classifier is not None:
                save_dict['classifier'] = eval_classifier.state_dict()
            torch.save(save_dict,
                       os.path.join(args.output_dir, f'checkpoint-{epoch}.pth'))

        # Save log
        log_stats = {
            'epoch': epoch, 'train_loss': avg_train_loss,
            'test_acc': test_stats['acc1'], 'test_loss': test_stats['loss'],
            'max_acc': max_accuracy, 'lr': cur_lr,
            'per_class_recall': {c: per_class[c]['recall'] for c in range(args.nb_classes)},
            'confusion_matrix': confusion.tolist(),
        }
        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write(json.dumps(log_stats) + "\n")

    # ---- Final ----
    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"\nTraining complete in {total_time}")
    print(f"Best accuracy: {max_accuracy:.2f}%")

    # Save final confusion matrix
    np.save(os.path.join(args.output_dir, "confusion_matrix_final.npy"),
            confusion.numpy())

    # Save final plots
    metrics.save_plots()

    # Save summary
    summary = {
        "mode": args.peft_mode,
        "dataset": args.data_set,
        "split_type": args.split_type,
        "n_clients": args.n_clients,
        "rounds": args.max_communication_rounds,
        "best_accuracy": max_accuracy,
        "trainable_params": n_trainable,
        "total_params": sum(p.numel() for p in model.parameters()),
        "lora_rank": args.lora_rank if 'lora' in args.peft_mode else None,
        "lora_targets": args.lora_targets if 'lora' in args.peft_mode else None,
        "training_time": total_time,
        "per_class_recall_final": {c: per_class[c]['recall'] for c in range(args.nb_classes)},
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
