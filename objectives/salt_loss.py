"""
objectives/salt_loss.py -- Learning objective for FedMamba-SALT.

The core loss is 1 - cosine_similarity between the student's projected
embedding and the teacher's embedding, averaged over the batch.

To prevent representation collapse (all embeddings converging to the same
point), a VICReg-style **variance penalty** is added.  The variance term
computes ``max(0, gamma - std(z_i))`` for each embedding dimension ``i``
and averages over dimensions.  This creates a hinge force that activates
*only* when per-dimension standard deviation drops below ``gamma``,
pushing embeddings apart without interfering when variance is healthy.

Total loss::

    L = L_align  +  lambda_var * L_var

where:
    L_align = 1 - cos_sim(student_proj, teacher_emb)
    L_var   = mean(max(0, gamma - std(student_proj, dim=batch)))

Implementation details:
    1. **Detach the teacher** -- explicit ``.detach()`` as safety net.
    2. **L2-normalise both vectors** before cosine for numerical stability.
    3. **Variance is on raw (unnormalized) student projections** -- computing
       std on L2-normalized vectors would hide collapse because
       normalization maps every vector to the unit sphere.

References:
    - BYOL (Grill et al., 2020) -- projection-head architectural pattern
    - VICReg (Bardes et al., 2022) -- variance-invariance-covariance
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
# VICReg variance penalty
# ======================================================================
def variance_loss(
    embeddings: torch.Tensor,
    gamma: float = 1.0,
) -> torch.Tensor:
    """
    VICReg-style variance hinge loss.

    For each embedding dimension, compute the standard deviation across
    the batch.  If std < gamma, apply a penalty of (gamma - std).
    Otherwise the penalty is zero.  Average over all dimensions.

    This creates a *repulsive force* that activates only when the
    representation is collapsing, without interfering with alignment
    when variance is already healthy.

    Args:
        embeddings: ``(B, D)`` raw (unnormalized) projected embeddings.
        gamma: Target minimum std per dimension (default 1.0 per VICReg).

    Returns:
        Scalar loss tensor (0.0 when all dimensions have std >= gamma).
    """
    # std across the batch for each dimension: shape (D,)
    std = embeddings.std(dim=0)
    # Hinge: penalise only when std < gamma
    return F.relu(gamma - std).mean()


# ======================================================================
# SALT loss function (with VICReg variance regularisation)
# ======================================================================
def salt_loss(
    student_proj: torch.Tensor,
    teacher_emb: torch.Tensor,
    lambda_var: float = 1.0,
    gamma: float = 1.0,
) -> tuple:
    """
    Combined alignment + variance loss for FedMamba-SALT.

    Total loss = L_align + lambda_var * L_var

    where:
        L_align = 1 - cosine_similarity   (range [0, 2])
        L_var   = VICReg variance hinge    (range [0, gamma])

    Args:
        student_proj: ``(B, D)`` output of the projection head.
        teacher_emb:  ``(B, D)`` CLS-token embedding from frozen teacher.
        lambda_var:   Weight for the variance penalty (default 1.0).
        gamma:        Target minimum per-dimension std (default 1.0).

    Returns:
        Tuple of ``(total_loss, align_loss, var_loss)`` -- all scalar
        tensors.  ``total_loss`` is the value to call ``.backward()`` on.
        The other two are for logging.
    """
    # --- Detach the teacher ---
    teacher_emb = teacher_emb.detach()

    # --- Variance loss on RAW (unnormalized) student projections ---
    # Must be computed BEFORE L2-normalization, because normalization
    # maps everything to the unit sphere and hides collapse.
    var_loss = variance_loss(student_proj, gamma=gamma)

    # --- L2-normalise for cosine alignment ---
    student_norm = F.normalize(student_proj, dim=-1, p=2)
    teacher_norm = F.normalize(teacher_emb, dim=-1, p=2)

    # --- Cosine alignment loss ---
    align_loss = (1.0 - F.cosine_similarity(student_norm, teacher_norm, dim=-1)).mean()

    # --- Combined ---
    total_loss = align_loss + lambda_var * var_loss

    return total_loss, align_loss, var_loss


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

