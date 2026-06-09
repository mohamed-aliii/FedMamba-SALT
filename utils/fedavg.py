
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
    global_model: nn.Module,
    client_models: List[nn.Module],
    client_class_counts: List[Dict[int, int]],
    num_classes: int
) -> None:
    """
    Performs true row-wise aggregation for the final linear layer.
    Each row `c` is aggregated only using clients that have samples for class `c`.
    The weight for client `k` on row `c` is proportional to its sample count: N_{k,c} / N_{global,c}.
    """
    global_state = global_model.state_dict()
    n_clients = len(client_models)
    
    # 1. Identify final linear layer keys
    final_weight_key, final_bias_key = find_final_linear_keys(global_state)
    
    # 2. Compute global totals for each class
    global_class_totals = {c: 0 for c in range(num_classes)}
    for counts in client_class_counts:
        for c, count in counts.items():
            global_class_totals[c] += count
            
    # 3. Aggregate final_weight_key
    global_weight = torch.zeros_like(global_state[final_weight_key], dtype=torch.float32)
    for c in range(num_classes):
        total_c = global_class_totals.get(c, 0)
        if total_c == 0:
            # Fallback if no client has this class (shouldn't happen in practice)
            # Just take uniform average
            for k in range(n_clients):
                client_weight = client_models[k].state_dict()[final_weight_key]
                global_weight[c] += client_weight[c].float() / n_clients
        else:
            for k in range(n_clients):
                count_kc = client_class_counts[k].get(c, 0)
                if count_kc > 0:
                    weight_fraction = count_kc / total_c
                    client_weight = client_models[k].state_dict()[final_weight_key]
                    global_weight[c] += weight_fraction * client_weight[c].float()
    
    global_state[final_weight_key] = global_weight
    
    # 4. Aggregate final_bias_key if it exists
    if final_bias_key is not None:
        global_bias = torch.zeros_like(global_state[final_bias_key], dtype=torch.float32)
        for c in range(num_classes):
            total_c = global_class_totals.get(c, 0)
            if total_c == 0:
                for k in range(n_clients):
                    client_bias = client_models[k].state_dict()[final_bias_key]
                    global_bias[c] += client_bias[c].float() / n_clients
            else:
                for k in range(n_clients):
                    count_kc = client_class_counts[k].get(c, 0)
                    if count_kc > 0:
                        weight_fraction = count_kc / total_c
                        client_bias = client_models[k].state_dict()[final_bias_key]
                        global_bias[c] += weight_fraction * client_bias[c].float()
        
        global_state[final_bias_key] = global_bias
        
    # 5. Load updated state back into global model
    global_model.load_state_dict(global_state)
