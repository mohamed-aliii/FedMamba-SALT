"""
objectives/salt_loss.py — Learning objective for FedMamba-SALT.

The loss is normalised MSE between the student's projected embedding and
the teacher's embedding.  Three implementation details are critical:

    1. **Detach the teacher** — ``teacher_emb.detach()`` is called inside the
       loss function as an explicit guarantee that no gradients flow into the
       teacher, even across PyTorch versions with different autograd semantics.

    2. **L2-normalise both vectors** — Without normalisation, the trivial
       minimum is the zero vector (representation collapse).  On the unit
       hypersphere the only way to reduce MSE is to align *directions*, not
       magnitudes.  Normalised MSE = 2·(1 − cos θ), ranging from 0
       (identical) to 4 (opposite).

    3. **Use F.mse_loss, not cosine embedding loss** — MSE gives cleaner
       gradient magnitudes and requires no margin hyperparameter.

References:
    • BYOL (Grill et al., 2020) — projection-head architectural pattern
    • VICReg (Bardes et al., 2022) — embedding std collapse diagnostic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# Projection Head (BYOL-style 3-layer MLP)
# ======================================================================
class ProjectionHead(nn.Module):
    """
    3-layer MLP that maps the student encoder's output to a projected
    embedding aligned with the teacher's representation space.

    Architecture::

        Linear(in_dim, hidden_dim)
        → BatchNorm1d(hidden_dim) → GELU
        → Linear(hidden_dim, hidden_dim)
        → BatchNorm1d(hidden_dim) → GELU
        → Linear(hidden_dim, out_dim)
        (no activation after the final linear layer)

    The projection head absorbs the geometric mismatch between the Mamba
    student and the ViT teacher without distorting either encoder's
    internal representations.
    """

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 2048,
        out_dim: int = 768,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, in_dim)`` student encoder output.

        Returns:
            ``(B, out_dim)`` projected embedding.
        """
        return self.net(x)


# ======================================================================
# SALT loss function
# ======================================================================
def salt_loss(
    student_proj: torch.Tensor,
    teacher_emb: torch.Tensor,
) -> torch.Tensor:
    """
    Normalised MSE loss between the student's projected embedding and the
    teacher's embedding.

    Args:
        student_proj: ``(B, D)`` output of the projection head.
        teacher_emb:  ``(B, D)`` CLS-token embedding from the frozen teacher.

    Returns:
        Scalar loss tensor.
    """
    # Detail 1 — Detach the teacher embedding.
    # The teacher's @torch.no_grad() prevents gradient computation through
    # teacher *parameters*, but does not fully sever the computation graph
    # in all PyTorch versions.  Explicit .detach() is the definitive
    # guarantee that no gradient signal flows into the teacher.
    teacher_emb = teacher_emb.detach()

    # Detail 2 — L2-normalise both vectors.
    # Constrains embeddings to the unit hypersphere where the only way to
    # reduce MSE is to align directions, not magnitudes.
    # On the unit sphere: MSE = 2 * (1 - cos(θ))
    #   • identical vectors → 0.0
    #   • orthogonal vectors → 2.0
    #   • opposite vectors   → 4.0
    student_proj = F.normalize(student_proj, dim=-1, p=2)
    teacher_emb = F.normalize(teacher_emb, dim=-1, p=2)

    # Detail 3 — Use F.mse_loss (not cosine embedding loss).
    # MSE gives cleaner gradient magnitudes and has no margin
    # hyperparameter to tune.
    return F.mse_loss(student_proj, teacher_emb)


# ======================================================================
# Collapse detection utility
# ======================================================================
def embedding_std(embeddings: torch.Tensor) -> float:
    """
    Compute the average standard deviation across embedding dimensions.

    A single scalar diagnostic for representation collapse:
        • Healthy:   > 0.1
        • Warning:   0.01 – 0.1
        • Collapsed: < 0.01

    Reference: VICReg (Bardes et al., 2022) uses this exact diagnostic.

    Args:
        embeddings: ``(B, D)`` batch of embedding vectors.

    Returns:
        Average per-dimension standard deviation (scalar float).
    """
    # std across the batch dimension for each embedding dimension,
    # then average over dimensions.
    return embeddings.std(dim=0).mean().item()
