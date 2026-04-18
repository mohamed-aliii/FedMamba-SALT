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
       so the variance penalty is a safety-net only.

Total loss::

    L = SmoothL1(student_proj, teacher_emb.detach()) + lambda_var * var_loss

Implementation details:
    1. **Detach the teacher** -- explicit ``.detach()`` as safety net.
    2. **No L2-normalisation** -- Smooth L1 operates on raw vectors
       so the student must match the teacher's actual geometry.
    3. **LayerNorm in projection head** -- federated-safe (no batch
       statistics dependency) while still centering/scaling the
       representation space for proper gradient flow.

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
        → LayerNorm(hidden_dim) → GELU
        → Linear(hidden_dim, hidden_dim)
        → LayerNorm(hidden_dim) → GELU
        → Linear(hidden_dim, out_dim)
        (no activation after the final linear layer)

    The projection head absorbs the geometric mismatch between the Mamba
    student and the ViT teacher without distorting either encoder's
    internal representations.  LayerNorm is used (not BatchNorm) so the
    module is federated-safe (no batch statistics dependency).
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
# Per-dimension variance penalty
# ======================================================================
def variance_loss(
    student_embeddings: torch.Tensor,
    teacher_embeddings: torch.Tensor,
) -> torch.Tensor:
    """
    Per-dimension variance penalty to prevent mean-target collapse
    without distorting anisotropic structural geometry.

    This creates a *repulsive force* that activates only when the
    representation is collapsing locally relative to the teacher's structure.

    Args:
        student_embeddings: ``(B, D)`` raw projected embeddings.
        teacher_embeddings: ``(B, D)`` detached teacher target embeddings.

    Returns:
        Scalar loss tensor.
    """
    # Calculate standard deviation for each dimension independently (shape: D)
    s_std = student_embeddings.std(dim=0)
    t_std = teacher_embeddings.std(dim=0)

    # Hinge: penalize the student only if a specific dimension drops below
    # the exact naturally occurring variance of the teacher in that same dimension.
    # Scaled by 0.9 to provide a smooth convergence floor.
    return F.relu((t_std * 0.9) - s_std).mean()


# ======================================================================
# SALT loss function (Smooth L1 manifold distillation)
# ======================================================================
def salt_loss(
    student_proj: torch.Tensor,
    teacher_emb: torch.Tensor,
    lambda_var: float = 1.0,
    gamma: float = 1.0,  # kept for signature compatibility
) -> tuple:
    """
    Manifold distillation loss for FedMamba-SALT.

    Total loss = SmoothL1(student_proj, teacher_emb) + lambda_var * var_loss

    Smooth L1 (Huber loss) matches both direction AND magnitude of the
    teacher's representation space.  The per-dimension variance penalty
    prevents mean-target collapse by enforcing that the student maintains
    at least 90% of the teacher's per-dimension variance.

    Args:
        student_proj: ``(B, D)`` output of the projection head.
        teacher_emb:  ``(B, D)`` embedding from frozen teacher.
        lambda_var:   Weight for the variance penalty (default 1.0).
        gamma:        Unused (kept for API compatibility).

    Returns:
        Tuple of ``(total_loss, align_loss, var_loss)``.
    """
    # --- Detach the teacher ---
    teacher_emb = teacher_emb.detach()

    # --- Direct Smooth L1 alignment (matches direction + magnitude) ---
    align_loss = F.smooth_l1_loss(student_proj, teacher_emb)

    # --- Strict Per-Dimension Variance Penalty ---
    var_loss = variance_loss(student_proj, teacher_emb)

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
