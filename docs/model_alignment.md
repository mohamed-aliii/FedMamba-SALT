# FedMamba-SALT Model Alignment

Reference model:

- Bingquan Huang, Yue Liu, Bin Tang, Gang Fang. "InceptionMamba: A Lightweight and Effective Model for Medical Image Classification Revealing Mamba's Low-Frequency Bias." Neural Processing Letters 58:15, 2026. DOI: 10.1007/s11063-025-11823-0.
- Official reference repository: https://github.com/pepper1329/InceptionMamba

## Preserved From InceptionMamba

`models/inception_mamba.py` keeps the core InceptionMamba block design:

- Split channels into an Inception local branch and an SSM global branch.
- Use four Inception paths: `1x1`, `1x1 -> 3x3`, `1x1 -> 3x3 -> 3x3`, and `AvgPool3x3 -> 1x1`.
- Apply channel attention after the multi-scale Inception fusion.
- Use an SSM path with `LayerNorm`, `Linear`, `DWConv3x3`, `SiLU`, four-direction scan logic, gating, and a final projection.
- Fuse the two branches with channel concatenation, channel shuffle, and a residual connection.

## FedMamba-SALT Adaptations

The full paper classifier backbone is intentionally adapted for this research:

- The paper uses `4x4` patch embedding, four stages, patch merging, and a classifier head.
- FedMamba-SALT uses `16x16` patches and no patch merging so the student preserves the same `14x14 = 196` patch grid as the frozen MAE ViT-B/16 teacher.
- The encoder outputs `768`-dimensional patch tokens to match the teacher representation for dense SALT distillation.
- `GroupNorm` replaces `BatchNorm2d` in convolution blocks to avoid client-specific running-stat drift during federated training.
- The paper classifier is replaced by the project-specific evaluation heads: linear probe, federated fine-tuning, attention pooling, and TTA evaluation.

## Parameter Accounting

Current paper experiment configuration:

- Encoder: `patch_size=16`, `embed_dim=448`, `depth=6`, `out_dim=768`.
- Encoder parameters: about `10.1M` with the local mock Mamba and about `10.9M` with real `mamba-ssm`.
- SALT projection head: about `7.35M` trainable parameters during pretraining.
- Frozen MAE ViT-B/16 teacher parameters are not counted as trainable.

This is the intended customized model: InceptionMamba block semantics from the paper, adapted to dense teacher-student federated representation learning.
