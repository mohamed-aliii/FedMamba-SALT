"""Teacher embedding statistics for SALT normalization diagnostics."""

from __future__ import annotations

from typing import Iterable

import torch


@torch.no_grad()
def compute_teacher_embedding_stats(
    teacher: torch.nn.Module,
    dataloaders: Iterable,
    device: str,
    max_batches: int = 0,
) -> dict:
    """
    Aggregate fixed teacher mean/std from first and second moments.

    The returned tensors are shaped like one teacher sample without the batch
    dimension, e.g. ``(196, 768)`` for dense patch distillation.
    """
    teacher.eval()
    sum_x = None
    sum_x2 = None
    count = 0
    batches_seen = 0

    for loader in dataloaders:
        for batch in loader:
            teacher_view = batch[0].to(device, non_blocking=True)
            emb = teacher(teacher_view, return_patches=True).detach().float()
            batch_sum = emb.sum(dim=0)
            batch_sum2 = emb.pow(2).sum(dim=0)
            if sum_x is None:
                sum_x = torch.zeros_like(batch_sum)
                sum_x2 = torch.zeros_like(batch_sum2)
            sum_x += batch_sum
            sum_x2 += batch_sum2
            count += emb.shape[0]
            batches_seen += 1
            if max_batches > 0 and batches_seen >= max_batches:
                break
        if max_batches > 0 and batches_seen >= max_batches:
            break

    if count <= 0 or sum_x is None or sum_x2 is None:
        raise ValueError("No teacher batches were available for statistic computation.")

    mean = sum_x / count
    var = (sum_x2 / count) - mean.pow(2)
    std = var.clamp_min(1e-12).sqrt()
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise FloatingPointError("Teacher statistic computation produced non-finite values.")

    return {
        "mean": mean.detach().cpu(),
        "std": std.detach().cpu(),
        "count": count,
        "batches": batches_seen,
    }


def teacher_stats_summary(stats: dict) -> dict:
    std = stats["std"].detach().float()
    mean = stats["mean"].detach().float()
    return {
        "count": int(stats.get("count", 0)),
        "batches": int(stats.get("batches", 0)),
        "mean_abs": mean.abs().mean().item(),
        "std_mean": std.mean().item(),
        "std_min": std.min().item(),
        "std_max": std.max().item(),
    }
