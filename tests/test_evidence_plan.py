"""Focused tests for evidence-gated COVID/general medical fixes."""

import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.linear_probe import FocalLoss, get_tta_transforms


def test_focal_loss_extreme_logits() -> bool:
    logits = torch.tensor(
        [[1000.0, -1000.0], [-1000.0, 1000.0], [0.0, 0.0]],
        requires_grad=True,
    )
    targets = torch.tensor([0, 1, 1])
    loss = FocalLoss(gamma=2.0)(logits, targets)
    loss.backward()
    ok = torch.isfinite(loss).item() and torch.isfinite(logits.grad).all().item()
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] FocalLoss finite for extreme logits: {loss.item():.6f}")
    return ok


def test_dataset_tta_presets() -> bool:
    covid_ok = len(get_tta_transforms("covidfl")) == 2
    retina_ok = len(get_tta_transforms("retina")) == 4
    ok = covid_ok and retina_ok
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] Dataset-aware TTA presets: covid=2 retina=4")
    return ok


if __name__ == "__main__":
    print("=" * 62)
    print("  Evidence Plan Tests")
    print("=" * 62)
    results = [
        test_focal_loss_extreme_logits(),
        test_classwise_aggregation_shared_layers_and_rows(),
        test_classwise_aggregation_preserves_absent_rows(),
        test_dataset_tta_presets(),
    ]
    print("=" * 62)
    print(f"  Total: {sum(results)}/{len(results)} passed")
    print("=" * 62)
    sys.exit(0 if all(results) else 1)
