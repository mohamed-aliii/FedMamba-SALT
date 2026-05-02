"""
utils/fedavg.py -- FedAvg / FedProx utilities for FedMamba-SALT.

Provides:
    - average_models:  Weighted average of client state_dicts into a global model.
    - proximal_loss:   FedProx proximal term  (mu/2) * ||w - w_global||^2.
                       Returns 0 when mu=0 (pure FedAvg mode).

Usage:
    # FedAvg:  mu=0.0  (no proximal term)
    # FedProx: mu=0.01 (typical starting value)

Reference:
    - FedAvg:  McMahan et al., "Communication-Efficient Learning of Deep Networks
               from Decentralized Data", AISTATS 2017.
    - FedProx: Li et al., "Federated Optimization in Heterogeneous Networks",
               MLSys 2020.
"""

import copy
from collections import OrderedDict
from typing import Dict, List, Optional

import torch
import torch.nn as nn


def average_models(
    global_model: nn.Module,
    client_models: List[nn.Module],
    client_weights: List[float],
) -> None:
    """
    Weighted average of client model parameters into global_model (in-place).

    After this call, global_model.state_dict() contains the weighted average
    of all client state_dicts.  The client models are NOT modified.

    Args:
        global_model:   Target model whose parameters will be overwritten.
        client_models:  List of N client models (same architecture as global).
        client_weights: List of N floats summing to 1.0 (typically proportional
                        to each client's dataset size).
    """
    global_state = global_model.state_dict()
    n_clients = len(client_models)

    # Zero out global state
    for key in global_state:
        global_state[key] = torch.zeros_like(global_state[key], dtype=torch.float32)

    # Weighted accumulation
    for client_idx in range(n_clients):
        client_state = client_models[client_idx].state_dict()
        w = client_weights[client_idx]
        for key in global_state:
            global_state[key] += w * client_state[key].float()

    # Load averaged parameters
    global_model.load_state_dict(global_state)


def broadcast_global_to_clients(
    global_model: nn.Module,
    client_models: List[nn.Module],
) -> None:
    """
    Copy global model parameters to all client models (in-place).

    Args:
        global_model:  Source model with averaged parameters.
        client_models: List of client models to overwrite.
    """
    global_state = global_model.state_dict()
    for client_model in client_models:
        client_model.load_state_dict(global_state)


def proximal_loss(
    local_model: nn.Module,
    global_params: Dict[str, torch.Tensor],
    mu: float = 0.01,
) -> torch.Tensor:
    """
    FedProx proximal term: (mu/2) * sum_i ||w_i - w_global_i||^2

    When mu=0, returns a zero tensor (pure FedAvg, no gradient overhead).

    Args:
        local_model:   Client model being trained.
        global_params: Snapshot of global model state_dict (detached).
        mu:            Proximal penalty strength.  0 = FedAvg, >0 = FedProx.

    Returns:
        Scalar proximal penalty to add to the SALT loss.
    """
    if mu <= 0:
        return torch.tensor(0.0, device=next(local_model.parameters()).device)

    penalty = torch.tensor(0.0, device=next(local_model.parameters()).device)
    for name, param in local_model.named_parameters():
        if param.requires_grad and name in global_params:
            penalty += (param - global_params[name].detach()).pow(2).sum()

    return (mu / 2.0) * penalty


def compute_client_weights(client_dataset_sizes: List[int]) -> List[float]:
    """
    Compute FedAvg aggregation weights proportional to dataset size.

    Args:
        client_dataset_sizes: List of N integers (number of samples per client).

    Returns:
        List of N floats summing to 1.0.
    """
    total = sum(client_dataset_sizes)
    return [s / total for s in client_dataset_sizes]
