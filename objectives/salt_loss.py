"""
objectives/salt_loss.py -- Learning objective for FedMamba-SALT.

The core loss is Smooth L1 (Huber) between the student's projected
embedding and the frozen teacher's embedding, averaged over the batch.

Smooth L1 was chosen over cosine similarity because:
    1. It matches both **direction and magnitude** of the teacher's
       representation space, preventing the angular-collapse trap where
       cosine similarity allows all vectors to point in one direction
       while scaling magnitude to satisfy a variance penalty.
    2. With a frozen teacher, there is no risk of joint collapse,
       so the VICReg variance penalty is unnecessary and was in fact
       creating a geometric paradox (forcing variance the teacher
       itself doesn't possess: t_std ≈ 0.04).

Total loss::

    L = SmoothL1(student_proj, teacher_emb.detach())

Implementation details:
    1. **Detach the teacher** -- explicit ``.detach()`` as safety net.
    2. **No L2-normalisation** -- Smooth L1 operates on raw vectors
       so the student must match the teacher's actual geometry.
    3. **LayerNorm in projection head** -- replaces BatchNorm1d which
       was laundering collapsed encoder outputs into fake variance.

References:
    - BYOL (Grill et al., 2020) -- projection-head architectural pattern
    - Smooth L1 / Huber loss -- robust regression standard
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
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
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
# SALT loss function (Smooth L1 direct manifold distillation)
# ======================================================================
def salt_loss(
    student_proj: torch.Tensor,
    teacher_emb: torch.Tensor,
    lambda_var: float = 1.0,
    gamma: float = 1.0,
) -> tuple:
    """
    Direct manifold distillation loss for FedMamba-SALT.

    Total loss = SmoothL1(student_proj, teacher_emb)

    Smooth L1 (Huber loss) matches both direction AND magnitude of the
    teacher's representation space. 

    Crucially, because the student sees heavily corrupted images while
    the teacher sees clean images, the network is prone to 'mean-target
    collapse', where the student just predicts the batch average to 
    minimize alignment error. To prevent this, we re-instate the VICReg
    variance penalty, but dynamically set `gamma` to the teacher's 
    actual standard deviation (instead of an incompatible 1.0).

    Args:
        student_proj: ``(B, D)`` output of the projection head.
        teacher_emb:  ``(B, D)`` CLS-token embedding from frozen teacher.
        lambda_var:   Weight for the variance penalty (default 1.0).
        gamma:        Scale override (if None, dynamic matching is used).

    Returns:
        Tuple of ``(total_loss, align_loss, var_loss)`` -- all scalar
        tensors.  ``total_loss`` is the value to call ``.backward()`` on.
        ``var_loss`` is always 0.0 (kept for logging compatibility).
    """
    # --- Detach the teacher ---
    teacher_emb = teacher_emb.detach()

    # --- Direct Smooth L1 alignment (matches direction + magnitude) ---
    align_loss = F.smooth_l1_loss(student_proj, teacher_emb)

    # --- Dynamic Variance Penalty ---
    # Computes the actual batch-level variance of the frozen target
    target_std = teacher_emb.std(dim=0).mean().item()
    dynamic_gamma = target_std if gamma is None or gamma == 1.0 else gamma
    
    # Penalize the student only if its representations collapse below the
    # structural variance of the teacher's embeddings.
    var_loss = variance_loss(student_proj, gamma=dynamic_gamma)

    # --- Total loss ---
    total_loss = align_loss + (lambda_var * var_loss)

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

