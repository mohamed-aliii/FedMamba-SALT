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
    server_momentum: Optional[Dict[str, torch.Tensor]] = None,
    beta: float = 0.9,
) -> None:
    """
    Weighted average of client model parameters into global_model (in-place).
    Supports FedAvgM (Server Momentum) if server_momentum dictionary is provided.

    Args:
        global_model:   Target model whose parameters will be overwritten.
        client_models:  List of N client models (same architecture as global).
        client_weights: List of N floats summing to 1.0.
        server_momentum: Optional dictionary storing server-side momentum buffers.
        beta:           Momentum coefficient (typically 0.9).
    """
    global_state = global_model.state_dict()
    n_clients = len(client_models)

    # Compute averaged weights in a temporary dictionary
    averaged_state = OrderedDict()
    for key in global_state:
        averaged_state[key] = torch.zeros_like(global_state[key], dtype=torch.float32)

    for client_idx in range(n_clients):
        client_state = client_models[client_idx].state_dict()
        w = client_weights[client_idx]
        for key in global_state:
            averaged_state[key] += w * client_state[key].float()

    if server_momentum is not None:
        # FedAvgM: Server-side momentum update
        for key in global_state:
            if global_state[key].is_floating_point():
                # Pseudo-gradient: Delta = Averaged_Weights - Global_Weights
                delta = averaged_state[key] - global_state[key]
                # Update momentum: m = beta * m + delta
                server_momentum[key] = beta * server_momentum[key] + delta
                # Update global weights: w = w + m
                global_state[key] = global_state[key] + server_momentum[key]
            else:
                # Non-floating point tensors (e.g., num_batches_tracked) bypass momentum
                global_state[key] = averaged_state[key]
    else:
        # Standard FedAvg
        for key in global_state:
            global_state[key] = averaged_state[key]

    # Load updated parameters back into the global model
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


def average_classifier_class_wise(
    global_classifier: nn.Module,
    client_classifiers: List[nn.Module],
    client_class_counts: List[Dict[int, int]],
    cls_weights: List[float],
    server_momentum: Optional[Dict[str, torch.Tensor]] = None,
    beta: float = 0.9,
) -> None:
    """
    Class-wise aggregation for the final classification layer to prevent 
    catastrophic drift from mono-class clients.
    Non-head layers are aggregated using cls_weights (sanitized) to prevent attention collapse.
    """
    global_state = global_classifier.state_dict()
    n_clients = len(client_classifiers)
    
    # 1. Identify the final linear layer keys
    final_weight_key = None
    final_bias_key = None
    for key in reversed(list(global_state.keys())):
        if key.endswith('.weight') and global_state[key].dim() == 2:
            final_weight_key = key
            bias_k = key.replace('.weight', '.bias')
            if bias_k in global_state:
                final_bias_key = bias_k
            break
            
    if not final_weight_key:
        raise ValueError("Could not identify final linear layer weight in classifier state_dict.")

    n_classes = global_state[final_weight_key].shape[0]

    # Calculate total global samples per class to find the denominator
    total_samples_per_class = torch.zeros(n_classes)
    for counts in client_class_counts:
        for cls_idx, count in counts.items():
            total_samples_per_class[cls_idx] += count
            
    # Compute averaged weights in a temporary dictionary
    averaged_state = OrderedDict()
    for key in global_state:
        averaged_state[key] = torch.zeros_like(global_state[key], dtype=torch.float32)

    for i in range(n_clients):
        client_state = client_classifiers[i].state_dict()
        w = cls_weights[i]
        
        for key in global_state:
            if key == final_weight_key or key == final_bias_key:
                # Handled below per-class
                continue
            if w > 0:
                averaged_state[key] += w * client_state[key].float()
            
        # Aggregate head per-class
        for cls_idx in range(n_classes):
            client_class_qty = client_class_counts[i].get(cls_idx, 0)
            if total_samples_per_class[cls_idx] > 0 and client_class_qty > 0:
                weight_ratio = client_class_qty / total_samples_per_class[cls_idx]
                
                # Weight
                averaged_state[final_weight_key][cls_idx] += client_state[final_weight_key][cls_idx].float() * weight_ratio
                # Bias
                if final_bias_key:
                    averaged_state[final_bias_key][cls_idx] += client_state[final_bias_key][cls_idx].float() * weight_ratio

    if server_momentum is not None:
        # FedAvgM: Server-side momentum update
        for key in global_state:
            if global_state[key].is_floating_point():
                delta = averaged_state[key] - global_state[key]
                server_momentum[key] = beta * server_momentum[key] + delta
                global_state[key] = global_state[key] + server_momentum[key]
            else:
                global_state[key] = averaged_state[key]
    else:
        # Standard FedAvg
        for key in global_state:
            global_state[key] = averaged_state[key]

    global_classifier.load_state_dict(global_state)
