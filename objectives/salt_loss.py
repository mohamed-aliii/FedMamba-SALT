"""
objectives/salt_loss.py -- Learning objective for FedMamba-SALT.

The core loss is Negative Cosine Similarity between the student's
embedding and the frozen teacher's embedding, averaged over the batch.

Cosine similarity is used because:
    1. It is **scale-invariant** -- it measures angular alignment on the
       unit sphere, entirely immune to the absolute magnitude of the
       teacher's representation space (~0.059 std).
    2. A student that predicts a constant (mean-target collapse) gets
       cosine similarity ~ 0 against varying teacher targets, making
       collapse a high-loss state rather than a trivial minimum.
    3. With a frozen teacher producing diverse per-image targets, there
       is no risk of joint angular collapse.

Total loss::

    L = mean(1 - cosine_similarity(student_emb, teacher_emb))

Implementation details:
    1. **Detach the teacher** -- explicit ``.detach()`` as safety net.
    2. **No L2-normalisation needed** -- cosine_similarity handles it.
    3. **No variance penalty needed** -- cosine loss on a frozen teacher
       inherently prevents collapse; variance terms add noise.

References:
    - BYOL (Grill et al., 2020) -- negative cosine similarity loss
    - DINO (Caron et al., 2021) -- self-distillation with frozen teacher
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# SALT loss function (Cosine Similarity distillation)
# ======================================================================
def salt_loss(
    student_proj: torch.Tensor,
    teacher_emb: torch.Tensor,
    lambda_var: float = 0.0,   # kept for signature compat, unused
    gamma: float = 0.0,        # kept for signature compat, unused
) -> tuple:
    """
    Cosine similarity distillation loss for FedMamba-SALT.

    Total loss = mean(1 - cosine_similarity(student, teacher))

    Range: [0, 2].  0 = perfect alignment, 1 = orthogonal, 2 = opposite.

    Args:
        student_proj: ``(B, D)`` output of the student encoder.
        teacher_emb:  ``(B, D)`` embedding from frozen teacher.
        lambda_var:   Unused (kept for API compatibility).
        gamma:        Unused (kept for API compatibility).

    Returns:
        Tuple of ``(total_loss, align_loss, var_loss)``.
        var_loss is always 0 (cosine loss needs no variance penalty).
    """
    # --- Detach the teacher ---
    teacher_emb = teacher_emb.detach()

    # --- Negative Cosine Similarity ---
    # F.cosine_similarity returns per-sample similarity in [-1, 1]
    # We want to minimize (1 - cos_sim), which is 0 at perfect alignment.
    align_loss = (1 - F.cosine_similarity(student_proj, teacher_emb, dim=-1)).mean()

    # No variance penalty needed -- cosine loss inherently prevents collapse
    var_loss = torch.tensor(0.0, device=student_proj.device)

    total_loss = align_loss

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
