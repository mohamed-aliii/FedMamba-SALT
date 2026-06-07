"""Focused tests for evidence-gated COVID/general medical fixes."""

import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.linear_probe import FocalLoss, get_tta_transforms
from utils.fedavg import average_classifier_class_wise


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


def _make_classifier(shared_value: float, row_values: list[float]) -> nn.Module:
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    with torch.no_grad():
        model[0].weight.fill_(shared_value)
        model[0].bias.fill_(shared_value)
        for row_idx, value in enumerate(row_values):
            model[2].weight[row_idx].fill_(value)
            model[2].bias[row_idx].fill_(value)
    return model


def test_classwise_aggregation_shared_layers_and_rows() -> bool:
    global_model = _make_classifier(0.0, [9.0, 9.0])
    client0 = _make_classifier(1.0, [10.0, 20.0])
    client1 = _make_classifier(3.0, [30.0, 40.0])

    average_classifier_class_wise(
        global_model,
        [client0, client1],
        client_class_counts=[{0: 5}, {1: 5}],
        cls_weights=[0.5, 0.5],
        shared_weights=[1.0, 0.0],
    )

    with torch.no_grad():
        shared_ok = torch.allclose(global_model[0].weight, client0[0].weight)
        row0_ok = torch.allclose(global_model[2].weight[0], client0[2].weight[0])
        row1_ok = torch.allclose(global_model[2].weight[1], client1[2].weight[1])
        bias_ok = (
            torch.allclose(global_model[2].bias[0], client0[2].bias[0])
            and torch.allclose(global_model[2].bias[1], client1[2].bias[1])
        )
    ok = shared_ok and row0_ok and row1_ok and bias_ok
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] Class-wise aggregation updates shared layers/rows as selected")
    return ok


def test_classwise_aggregation_preserves_absent_rows() -> bool:
    global_model = _make_classifier(0.0, [9.0, 7.0])
    client0 = _make_classifier(1.0, [10.0, 20.0])

    average_classifier_class_wise(
        global_model,
        [client0],
        client_class_counts=[{0: 5}],
        cls_weights=[1.0],
        shared_weights=[1.0],
    )

    with torch.no_grad():
        row0_ok = torch.allclose(global_model[2].weight[0], client0[2].weight[0])
        row1_ok = torch.allclose(global_model[2].weight[1], torch.full_like(global_model[2].weight[1], 7.0))
        bias1_ok = torch.allclose(global_model[2].bias[1], torch.tensor(7.0))
    ok = row0_ok and row1_ok and bias1_ok
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] Class-wise aggregation preserves rows for absent classes")
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
