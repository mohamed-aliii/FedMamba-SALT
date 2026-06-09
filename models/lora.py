import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.linear = linear_layer
        self.rank = rank
        self.scaling = alpha / rank
        device = linear_layer.weight.device
        
        # LoRA A and B matrices
        self.lora_A = nn.Parameter(torch.zeros(linear_layer.in_features, rank, device=device))
        self.lora_B = nn.Parameter(torch.zeros(rank, linear_layer.out_features, device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base_out = self.linear(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base_out + lora_out

class LoRAConv2d(nn.Module):
    def __init__(self, conv_layer: nn.Conv2d, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.conv = conv_layer
        self.rank = rank
        self.scaling = alpha / rank
        
        # LoRA for 1x1 convolutions
        assert conv_layer.kernel_size == (1, 1) or conv_layer.kernel_size == 1, "LoRAConv2d tailored for 1x1 convs here"
        
        device = conv_layer.weight.device
        self.lora_A = nn.Parameter(torch.zeros(rank, conv_layer.in_channels, 1, 1, device=device))
        self.lora_B = nn.Parameter(torch.zeros(conv_layer.out_channels, rank, 1, 1, device=device))
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base_out = self.conv(x)
        lora_out = F.conv2d(x, self.lora_A)
        lora_out = F.conv2d(lora_out, self.lora_B)
        return base_out + lora_out * self.scaling

def inject_lora_into_encoder(encoder: nn.Module, rank: int = 8, alpha: float = 16.0):
    """Recursively replaces target Linear/Conv1x1 layers with LoRA wrappers."""
    # 1. Freeze entire backbone
    for param in encoder.parameters():
        param.requires_grad = False
        
    # 2. Inject LoRA into critical projections and 1x1 bottlenecks
    targets = []
    for name, module in encoder.named_modules():
        if isinstance(module, nn.Linear) and any(
            target in name for target in ['proj_main', 'proj_gate', 'proj_out']
        ):
            targets.append((name, module, 'linear'))
        elif isinstance(module, nn.Conv2d) and module.kernel_size == (1, 1) and 'inception' in name:
            targets.append((name, module, 'conv'))
            
    for name, module, layer_type in targets:
        parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
        child_name = name.rsplit('.', 1)[-1] if '.' in name else name
        parent = encoder.get_submodule(parent_name) if parent_name else encoder
        
        if layer_type == 'linear':
            setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha))
        else:
            setattr(parent, child_name, LoRAConv2d(module, rank=rank, alpha=alpha))
            
    # 3. Unfreeze LayerNorms and GroupNorms for stability (standard practice in PEFT)
    for name, param in encoder.named_parameters():
        if "norm" in name or "LayerNorm" in name or "GroupNorm" in name:
            param.requires_grad = True
