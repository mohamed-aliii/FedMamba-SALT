"""
objectives/salt_loss.py -- Learning objective for FedMamba-SALT.

The core loss is **cosine similarity** between the student's projected
embedding (L2-normalised) and the frozen teacher's embedding
(L2-normalised), averaged over the batch.

Cosine similarity was chosen because the diagnostic analysis proved that:
    1. Class-discriminative information in the teacher lives in the
       **direction** of embedding vectors (centered cosine sim = -1.0
       between class centroids → perfect angular separation).
    2. Using SmoothL1 (magnitude-matching) caused the projection head
       to converge to a mean-prediction strategy that crushed class
       separation from 0.295 → 0.083 L2 (3.6× compression), rendering
       features random for linear probing.
    3. Cosine similarity focuses gradient budget entirely on angular
       alignment, preserving the directional class signal.

Total loss::

    L = (1 - cos_sim(normalise(s_proj), normalise(t_emb)))
        + lambda_var * var_loss(s_emb)

Implementation details:
    1. **L2-normalise before loss** -- both student projection and
       teacher embedding are unit-normalised so the loss operates
       purely on direction.
    2. **Variance penalty on ENCODER output (s_emb), NOT projector output**
       -- the projection head's LayerNorm rescales any input, masking
       encoder collapse. The penalty must guard the encoder directly.
    3. **LayerNorm in projection head** -- federated-safe (no batch
       statistics dependency).

References:
    - BYOL (Grill et al., 2020) -- projection-head architectural pattern
    - DINO (Caron et al., 2021) -- cosine distillation for SSL
    - VICReg (Bardes et al., 2022) -- variance regularisation
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
# Per-dimension variance penalty (applied to ENCODER output)
# ======================================================================
def variance_loss(
    encoder_embeddings: torch.Tensor,
    target_std: float = 0.1,
) -> torch.Tensor:
    """
    Per-dimension variance penalty to prevent encoder collapse.

    Penalises any dimension whose batch standard deviation drops below
    ``target_std``.  Applied to the RAW encoder output (before the
    projection head) to directly guard the encoder's representation
    quality.

    The projection head's LayerNorm rescales any input, so monitoring
    the projector output masks encoder collapse.  This penalty operates
    upstream at the encoder.

    Args:
        encoder_embeddings: ``(B, D)`` raw encoder output (before proj head).
        target_std: Minimum per-dimension std to maintain (default 0.1).

    Returns:
        Scalar loss tensor.
    """
    # Per-dimension std across the batch (shape: D)
    s_std = encoder_embeddings.std(dim=0)

    # Hinge: penalise only when std drops below target
    return F.relu(target_std - s_std).mean()


# ======================================================================
# SALT loss function (Cosine distillation)
# ======================================================================
def salt_loss(
    student_proj: torch.Tensor,
    teacher_emb: torch.Tensor,
    student_emb: torch.Tensor = None,
    lambda_var: float = 1.0,
) -> tuple:
    """
    Directional distillation loss for FedMamba-SALT.

    Total loss = (1 - cos_sim(norm(s_proj), norm(t_emb)))
                 + lambda_var * var_loss(s_emb)

    The cosine similarity focuses gradient entirely on angular alignment
    — the subspace where class-discriminative information lives.  The
    variance penalty prevents encoder collapse independently of the
    projection head.

    Args:
        student_proj: ``(B, D)`` output of the projection head.
        teacher_emb:  ``(B, D)`` embedding from frozen teacher.
        student_emb:  ``(B, D)`` raw encoder output (before proj head).
                      If None, variance penalty is skipped (for testing).
        lambda_var:   Weight for the variance penalty (default 1.0).

    Returns:
        Tuple of ``(total_loss, align_loss, var_loss)``.
    """
    # --- Detach the teacher ---
    teacher_emb = teacher_emb.detach()

    # --- L2-normalise both vectors ---
    s_norm = F.normalize(student_proj, dim=-1, p=2)
    t_norm = F.normalize(teacher_emb, dim=-1, p=2)

    # --- Cosine alignment loss (range [0, 2]) ---
    # cos_sim = (s_norm * t_norm).sum(dim=-1)  → range [-1, 1]
    # loss = 1 - cos_sim                        → range [0, 2]
    align_loss = (1.0 - (s_norm * t_norm).sum(dim=-1)).mean()

    # --- Variance penalty on ENCODER output (not projector) ---
    if student_emb is not None:
        var_loss = variance_loss(student_emb)
    else:
        var_loss = torch.tensor(0.0, device=student_proj.device)

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
