
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
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


import torch.nn.functional as F

def aggregate_prototypes_ema(
    global_centroids: Dict[int, torch.Tensor],
    client_local_centroids: List[Dict[int, torch.Tensor]],
    client_class_counts: List[Dict[int, int]],
    num_classes: int,
    device: torch.device,
    feat_dim: int = 768,
    momentum: float = 0.9,
    comm_round: int = 0
) -> Dict[int, torch.Tensor]:
    """
    Computes the weighted average of newly discovered local prototypes,
    then applies an Exponential Moving Average (EMA) against the existing 
    global prototypes to stabilize the representation space.
    
    Args:
        global_centroids: Dict of current global prototypes (can be empty for Round 0).
        client_local_centroids: List of dicts containing each client's local prototypes.
        client_class_counts: List of dicts containing sample counts per class.
        num_classes: Total number of classes.
        device: Torch device.
        feat_dim: Dimensionality of the representation space.
        momentum: EMA factor (alpha). 0.0 means complete overwrite, 1.0 means frozen.
        
    Returns:
        Updated dictionary of global centroids, L2-normalized.
    """
    new_global_centroids = {}
    
    for c in range(num_classes):
        # Find how many samples of class 'c' exist across all clients this round
        total_c_samples = sum(counts.get(c, 0) for counts in client_class_counts)
        
        if total_c_samples > 0:
            c_sum = torch.zeros(feat_dim, device=device)
            
            # Step 1: Weighted average of the NEW local prototypes
            for client_idx in range(len(client_local_centroids)):
                if c in client_local_centroids[client_idx]:
                    weight = client_class_counts[client_idx].get(c, 0) / total_c_samples
                    c_sum += client_local_centroids[client_idx][c].to(device) * weight
                    
            # Normalize the newly aggregated prototype
            new_c_normalized = F.normalize(c_sum, p=2, dim=0)
            
            # Step 2: EMA Update against the historical global prototype
            adaptive_momentum = momentum if comm_round >= 20 else max(0.0, momentum - 0.2)
            if global_centroids is not None and c in global_centroids:
                # Smooth shift
                updated_c = (adaptive_momentum * global_centroids[c].to(device)) + ((1.0 - adaptive_momentum) * new_c_normalized)
            else:
                # Cold start: No historical prototype exists yet
                updated_c = new_c_normalized
                
            # Step 3: Re-project to unit sphere to maintain Cosine Similarity validity
            new_global_centroids[c] = F.normalize(updated_c, p=2, dim=0)
            
        elif global_centroids is not None and c in global_centroids:
            # Edge case: No client had this class this round, keep historical
            new_global_centroids[c] = global_centroids[c].to(device)
            
    return new_global_centroids


def model_update_norm(global_model: nn.Module, client_model: nn.Module) -> float:
    """L2 norm of a client's floating-point update relative to the global model."""
    global_state = global_model.state_dict()
    client_state = client_model.state_dict()
    total = torch.tensor(0.0)
    for key, global_value in global_state.items():
        if key not in client_state or not global_value.is_floating_point():
            continue
        delta = client_state[key].detach().float().cpu() - global_value.detach().float().cpu()
        total += delta.pow(2).sum()
    return total.sqrt().item()


def average_models(
    global_model: nn.Module,
    client_models: List[nn.Module],
    client_weights: List[float],
    server_momentum: Optional[Dict[str, torch.Tensor]] = None,
    beta: float = 0.9,
) -> None:
    """
    Weighted average of client model parameters into global_model (in-place).
    Optimized: Only averages parameters that require gradients.
    """
    global_state = global_model.state_dict()
    n_clients = len(client_models)

    # Filter to only aggregate trainable parameters
    trainable_keys = {name for name, param in global_model.named_parameters() if param.requires_grad}

    # Compute averaged weights in a temporary dictionary
    averaged_state = OrderedDict()
    for key in trainable_keys:
        averaged_state[key] = torch.zeros_like(global_state[key], dtype=torch.float32)

    for client_idx in range(n_clients):
        client_state = client_models[client_idx].state_dict()
        w = client_weights[client_idx]
        for key in trainable_keys:
            averaged_state[key] += w * client_state[key].float()

    if server_momentum is not None:
        # FedAvgM: Server-side momentum update
        for key in trainable_keys:
            if global_state[key].is_floating_point():
                delta = averaged_state[key] - global_state[key]
                server_momentum[key] = beta * server_momentum[key] + delta
                global_state[key] = global_state[key] + server_momentum[key]
            else:
                global_state[key] = averaged_state[key]
    else:
        # Standard FedAvg
        for key in trainable_keys:
            global_state[key] = averaged_state[key]

    # Load updated parameters back into the global model
    global_model.load_state_dict(global_state)


def broadcast_global_to_clients(
    global_model: nn.Module,
    client_models: List[nn.Module],
) -> None:
    """
    Copy global model parameters to all client models (in-place).
    Optimized: Only broadcasts parameters that require gradients to save overhead.
    """
    global_state = global_model.state_dict()
    trainable_keys = {name for name, param in global_model.named_parameters() if param.requires_grad}
    
    for client_model in client_models:
        client_state = client_model.state_dict()
        for key in trainable_keys:
            client_state[key].copy_(global_state[key])


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
            diff = param - global_params[name].detach()
            # Orthogonal State Transition Regularization:
            # Apply a 10x stronger penalty to A_log to preserve the pre-trained ODE dynamics
            if 'A_log' in name:
                penalty += 10.0 * diff.pow(2).sum()
            else:
                penalty += diff.pow(2).sum()

    return (mu / 2.0) * penalty


def compute_client_weights(client_dataset_sizes: List[int], strategy: str = "size") -> List[float]:
    """
    Compute FedAvg aggregation weights.

    Args:
        client_dataset_sizes: List of N integers (number of samples per client).
        strategy: 'size' for proportional weighting, 'equal' for uniform weighting.

    Returns:
        List of N floats summing to 1.0.
    """
    n_clients = len(client_dataset_sizes)
    if strategy == "equal":
        return [1.0 / n_clients for _ in range(n_clients)]
        
    total = sum(client_dataset_sizes)
    return [s / total for s in client_dataset_sizes]


