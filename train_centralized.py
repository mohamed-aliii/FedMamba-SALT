"""
train_centralized.py — Main entry-point for centralized SALT training.

Trains a lightweight Inception-Mamba student encoder to match the embedding
space of a frozen MAE-pretrained ViT-B/16 teacher using asymmetric
augmentation as the learning signal.

Usage:
    python train_centralized.py --config configs/default.yaml
"""


def main() -> None:
    """Placeholder for the centralized training loop."""
    raise NotImplementedError(
        "Training loop not yet implemented. "
        "See implementation_plan.md for the roadmap."
    )


if __name__ == "__main__":
    main()
