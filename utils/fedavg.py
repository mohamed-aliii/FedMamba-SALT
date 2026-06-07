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


def find_final_linear_keys(state_dict: Dict[str, torch.Tensor]) -> Tuple[str, Optional[str]]:
    """Return the final linear weight/bias keys from a classifier state_dict."""
    final_weight_key = None
    final_bias_key = None
    for key in reversed(list(state_dict.keys())):
        if key.endswith(".weight") and state_dict[key].dim() == 2:
            final_weight_key = key
            bias_key = key.replace(".weight", ".bias")
            if bias_key in state_dict:
                final_bias_key = bias_key
            break
    if final_weight_key is None:
        raise ValueError("Could not identify final linear layer weight in classifier state_dict.")
    return final_weight_key, final_bias_key


def classifier_head_diagnostics(classifier: nn.Module) -> Dict[str, List[float]]:
    """Return row-wise weight norms and bias values for the final classifier head."""
    state = classifier.state_dict()
    final_weight_key, final_bias_key = find_final_linear_keys(state)
    weight = state[final_weight_key].detach().float()
    diag = {
        "final_weight_key": [final_weight_key],
        "row_weight_norms": weight.norm(dim=1).cpu().tolist(),
    }
    if final_bias_key is not None:
        bias = state[final_bias_key].detach().float()
        diag["final_bias_key"] = [final_bias_key]
        diag["row_biases"] = bias.cpu().tolist()
        diag["bias_abs_mean"] = [bias.abs().mean().item()]
    else:
        diag["final_bias_key"] = []
        diag["row_biases"] = []
        diag["bias_abs_mean"] = [0.0]
    return diag


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


def average_classifier_class_wise(
    global_classifier: nn.Module,
    client_classifiers: List[nn.Module],
    client_class_counts: List[Dict[int, int]],
    cls_weights: List[float],
    shared_weights: Optional[List[float]] = None,
    server_momentum: Optional[Dict[str, torch.Tensor]] = None,
    beta: float = 0.9,
) -> None:
    """
    Class-wise aggregation for the final classification layer to prevent
    catastrophic drift from clients that do not contain all classes.

    Args:
        cls_weights: Per-client weights used only for final class rows. A row
            receives contributions from clients that contain that class,
            normalized by per-class sample counts.
        shared_weights: Optional per-client weights for non-final classifier
            layers. Defaults to cls_weights to preserve the previous behavior.
    """
    global_state = global_classifier.state_dict()
    n_clients = len(client_classifiers)
    
    # 1. Identify the final linear layer keys
    final_weight_key, final_bias_key = find_final_linear_keys(global_state)

    n_classes = global_state[final_weight_key].shape[0]
    if shared_weights is None:
        shared_weights = cls_weights
    if len(shared_weights) != n_clients:
        raise ValueError("shared_weights length must match client_classifiers length.")

    # Calculate total global samples per class to find the denominator
    total_samples_per_class = torch.zeros(
        n_classes, device=global_state[final_weight_key].device,
    )
    for counts in client_class_counts:
        for cls_idx, count in counts.items():
            total_samples_per_class[cls_idx] += count
            
    # Compute averaged weights in a temporary dictionary
    averaged_state = OrderedDict()
    for key in global_state:
        if key == final_weight_key or key == final_bias_key:
            # Preserve rows for classes absent from every participating client.
            averaged_state[key] = global_state[key].detach().float().clone()
        elif global_state[key].is_floating_point():
            averaged_state[key] = torch.zeros_like(global_state[key], dtype=torch.float32)
        else:
            averaged_state[key] = global_state[key].clone()

    present_classes = total_samples_per_class > 0
    averaged_state[final_weight_key][present_classes] = 0.0
    if final_bias_key:
        averaged_state[final_bias_key][present_classes] = 0.0

    shared_total = float(sum(shared_weights))
    normalized_shared_weights = (
        [float(w) / shared_total for w in shared_weights]
        if shared_total > 0 else [0.0 for _ in shared_weights]
    )
    if shared_total <= 0:
        for key in global_state:
            if key != final_weight_key and key != final_bias_key and global_state[key].is_floating_point():
                averaged_state[key] = global_state[key].detach().float().clone()

    for i in range(n_clients):
        client_state = client_classifiers[i].state_dict()
        w = normalized_shared_weights[i]
        
        for key in global_state:
            if key == final_weight_key or key == final_bias_key:
                # Handled below per-class
                continue
            if w > 0 and global_state[key].is_floating_point():
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
