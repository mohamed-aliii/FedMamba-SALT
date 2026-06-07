import os

file_path = 'models/inception_mamba.py'
with open(file_path, 'r') as f:
    lines = f.readlines()

out = []
skip = False
for line in lines:
    if 'class PatchEmbedding(nn.Module):' in line:
        skip = True
    if not skip:
        out.append(line)

new_code = '''class PatchMerging(nn.Module):
    """Halves spatial resolution and doubles channels."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        import math
        H = W = int(math.isqrt(L))
        x_2d = x.transpose(1, 2).reshape(B, C, H, W)
        x_merged = self.proj(x_2d).flatten(2).transpose(1, 2)
        return self.norm(x_merged)


class PatchEmbedding(nn.Module):
    """Non-overlapping patch projection with LayerNorm."""
    def __init__(self, patch_size: int = 4, embed_dim: int = 96):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)
        return self.norm(x)


class InceptionMambaEncoder(nn.Module):
    """
    Paper-accurate hierarchical encoder: 4 stages with Patch Merging.
    """
    def __init__(
        self,
        patch_size: int = 4,
        embed_dim: int = 96,
        depths: list[int] = [2, 2, 4, 2],
        dims: list[int] = [96, 192, 384, 768],
        out_dim: int = 768,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_size, embed_dim)

        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        num_stages = len(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        cur = 0
        for i in range(num_stages):
            stage = nn.ModuleList([
                InceptionMambaBlock(dims[i], drop_path=dpr[cur + j])
                for j in range(depths[i])
            ])
            self.stages.append(stage)
            cur += depths[i]
            
            if i < num_stages - 1:
                self.downsamples.append(PatchMerging(dims[i], dims[i+1]))
            else:
                self.downsamples.append(nn.Identity())

        self.norm = nn.LayerNorm(dims[-1])
        self.proj = nn.Linear(dims[-1], out_dim)

    def forward(self, x: torch.Tensor, return_patches: bool = False,
                mask_ratio: float = 0.0) -> torch.Tensor:
        x = self.patch_embed(x)

        if mask_ratio > 0.0 and self.training:
            B, L, D = x.shape
            mask = torch.rand(B, L, device=x.device) < mask_ratio
            x = torch.where(mask.unsqueeze(-1), torch.zeros_like(x), x)

        for i in range(len(self.stages)):
            for blk in self.stages[i]:
                x = blk(x)
            if i < len(self.stages) - 1:
                x = self.downsamples[i](x)

        x = self.norm(x)
        x = self.proj(x)

        if return_patches:
            return x

        return x.mean(dim=1)
'''

out.append(new_code)
with open(file_path, 'w') as f:
    f.writelines(out)
print('Updated models/inception_mamba.py')
