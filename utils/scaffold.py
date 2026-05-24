"""
utils/scaffold.py -- SCAFFOLD algorithm for FedMamba-SALT.

SCAFFOLD (Stochastic Controlled Averaging for Federated Learning)
uses control variates to correct client gradient drift, providing
variance reduction that is provably better than FedAvg/FedProx
under data heterogeneity.

Reference:
    Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging
    for Federated Learning", ICML 2020.
"""

from collections import OrderedDict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


def init_control_variates(
    encoder: nn.Module,
    classifier: nn.Module,
    n_clients: int,
) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
    """
    Initialize server and per-client control variates to zero.

    Returns:
        c_global:  Server control variate {param_name: zero tensor}.
        c_clients: List of n_clients client control variates.
    """
    c_global = OrderedDict()
    for name, param in encoder.named_parameters():
        if param.requires_grad:
            c_global[f"enc.{name}"] = torch.zeros_like(param.data)
    for name, param in classifier.named_parameters():
        if param.requires_grad:
            c_global[f"cls.{name}"] = torch.zeros_like(param.data)

    c_clients = [
        OrderedDict({k: v.clone() for k, v in c_global.items()})
        for _ in range(n_clients)
    ]
    return c_global, c_clients


def apply_scaffold_correction(
    encoder: nn.Module,
    classifier: nn.Module,
    c_global: Dict[str, torch.Tensor],
    c_local: Dict[str, torch.Tensor],
) -> None:
    """
    Apply SCAFFOLD gradient correction in-place after backward().

    Modifies gradients: g_corrected = g + (c_global - c_local)
    """
    for name, param in encoder.named_parameters():
        key = f"enc.{name}"
        if param.grad is not None and key in c_global:
            param.grad.data.add_(c_global[key] - c_local[key])

    for name, param in classifier.named_parameters():
        key = f"cls.{name}"
        if param.grad is not None and key in c_global:
            param.grad.data.add_(c_global[key] - c_local[key])


def compute_control_variate_update(
    encoder: nn.Module,
    classifier: nn.Module,
    c_global: Dict[str, torch.Tensor],
    c_local: Dict[str, torch.Tensor],
    x_global: Dict[str, torch.Tensor],
    K: int,
    lr_enc: float,
    lr_cls: float,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Option II control variate update (from SCAFFOLD paper).

        c_i+ = c_i - c + (x_global - x_local) / (K * eta)

    Args:
        encoder, classifier: Client models AFTER local training.
        c_global: Server control variate.
        c_local:  Client's current control variate.
        x_global: Snapshot of global params BEFORE local training.
        K:        Total local gradient steps (E_epoch * batches_per_client).
        lr_enc:   Encoder learning rate for this round.
        lr_cls:   Classifier learning rate for this round.

    Returns:
        c_local_new: Updated client control variate.
        delta_c:     Change (c_new - c_old) for server update.
    """
    c_local_new = OrderedDict()
    delta_c = OrderedDict()

    for name, param in encoder.named_parameters():
        key = f"enc.{name}"
        if param.requires_grad and key in c_global:
            c_new = (
                c_local[key]
                - c_global[key]
                + (x_global[key] - param.data) / (K * lr_enc)
            )
            c_local_new[key] = c_new
            delta_c[key] = c_new - c_local[key]

    for name, param in classifier.named_parameters():
        key = f"cls.{name}"
        if param.requires_grad and key in c_global:
            c_new = (
                c_local[key]
                - c_global[key]
                + (x_global[key] - param.data) / (K * lr_cls)
            )
            c_local_new[key] = c_new
            delta_c[key] = c_new - c_local[key]

    return c_local_new, delta_c


def update_server_control_variate(
    c_global: Dict[str, torch.Tensor],
    all_delta_c: List[Dict[str, torch.Tensor]],
    n_clients: int,
) -> None:
    """
    Server control variate update (in-place).

        c+ = c + (1/N) * sum(delta_c_i)
    """
    for key in c_global:
        total_delta = torch.zeros_like(c_global[key])
        for delta_c in all_delta_c:
            if key in delta_c:
                total_delta.add_(delta_c[key])
        c_global[key].add_(total_delta / n_clients)
