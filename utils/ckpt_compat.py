"""
utils/ckpt_compat.py -- Cross-version compatible checkpoint loading.

SSL-FL checkpoints embed ``argparse.Namespace`` objects that contain
pandas DataFrames (``args.record_val_acc``, ``args.record_test_acc``).
When loaded on a newer pandas version, ``torch.load`` crashes because
the internal ``new_block()`` constructor changed its ``placement``
argument from ``slice`` to ``BlockPlacement``.

This module patches pandas before loading so the unpickler succeeds.
The ``'model'`` key (actual weights) is never affected.
"""

import torch


def _patch_pandas() -> None:
    """
    Monkey-patch ``pandas.core.internals.blocks.new_block`` so that
    old pickled DataFrames (which pass a plain ``slice`` for placement)
    can be unpickled on newer pandas versions that expect a
    ``BlockPlacement`` object.

    Safe to call multiple times -- only patches once.
    """
    try:
        import pandas as pd
        from pandas.core.internals import blocks as _blocks
    except ImportError:
        return  # no pandas => no problem

    # Check if already patched
    if getattr(_blocks, '_fedmamba_patched', False):
        return

    _original_new_block = _blocks.new_block

    def _patched_new_block(*args, **kwargs):
        # placement can be 3rd positional arg or a keyword arg
        from pandas._libs.internals import BlockPlacement
        if len(args) >= 3 and isinstance(args[2], slice):
            args = list(args)
            args[2] = BlockPlacement(args[2])
            args = tuple(args)
        elif 'placement' in kwargs and isinstance(kwargs['placement'], slice):
            kwargs['placement'] = BlockPlacement(kwargs['placement'])
        return _original_new_block(*args, **kwargs)

    _blocks.new_block = _patched_new_block
    _blocks._fedmamba_patched = True


def safe_torch_load(path: str, map_location: str = "cpu") -> dict:
    """
    Load a PyTorch checkpoint with graceful handling of stale pickled
    pandas DataFrames (found in SSL-FL checkpoints).

    Patches pandas internals before loading, then uses standard
    ``torch.load(..., weights_only=False)``.

    Args:
        path: Path to the ``.pth`` checkpoint file.
        map_location: Device mapping (default ``"cpu"``).

    Returns:
        The checkpoint dictionary.
    """
    _patch_pandas()
    return torch.load(path, map_location=map_location, weights_only=False)
