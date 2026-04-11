"""
utils/ckpt_compat.py -- Cross-version compatible checkpoint loading.

SSL-FL checkpoints embed ``argparse.Namespace`` objects that contain
pandas DataFrames (``args.record_val_acc``, ``args.record_test_acc``).
When loaded on a newer pandas, the ``new_block()`` constructor fails
because old pickled blocks pass ``slice`` for ``placement``, while
newer pandas expects ``BlockPlacement``.

This module hooks into torch.load's pickle machinery to intercept
``pandas.core.internals.blocks.new_block`` at unpickle time and patch
the placement argument before it reaches the C extension.
"""

import pickle
from typing import Any

import torch


def _make_safe_new_block():
    """
    Return a wrapper around ``pandas.core.internals.blocks.new_block``
    that converts ``slice`` placement args to ``BlockPlacement``.
    """
    from pandas.core.internals.blocks import new_block as _real_new_block
    from pandas._libs.internals import BlockPlacement

    def _safe_new_block(*args, **kwargs):
        # Convert slice -> BlockPlacement in positional args
        args = list(args)
        for i, a in enumerate(args):
            if isinstance(a, slice):
                args[i] = BlockPlacement(a)
        # Convert slice -> BlockPlacement in keyword args
        for k, v in kwargs.items():
            if isinstance(v, slice):
                kwargs[k] = BlockPlacement(v)
        return _real_new_block(*args, **kwargs)

    return _safe_new_block


class _CompatUnpickler(pickle.Unpickler):
    """
    Custom unpickler that intercepts ``pandas.core.internals.blocks.new_block``
    and returns a safe wrapper that converts slice -> BlockPlacement.

    This is the ONLY reliable way to handle this because:
    - Pickle's GLOBAL opcode calls ``find_class`` to resolve functions
    - The resolved function is then called via REDUCE with arguments
    - Monkeypatching the module attribute doesn't work if the pickle
      stream has a direct GLOBAL reference to the original function
    """

    def find_class(self, module: str, name: str) -> Any:
        if module == "pandas.core.internals.blocks" and name == "new_block":
            return _make_safe_new_block()
        return super().find_class(module, name)


class _CompatPickleModule:
    """
    A module-like object that provides our custom Unpickler to torch.load.
    torch.load uses ``pickle_module.Unpickler`` internally.
    """
    Unpickler = _CompatUnpickler

    # Forward all other pickle attributes to the real module
    def __getattr__(self, name):
        return getattr(pickle, name)


# Singleton instance
_compat_pickle = _CompatPickleModule()


def safe_torch_load(path: str, map_location: str = "cpu") -> dict:
    """
    Load a PyTorch checkpoint with graceful handling of stale pickled
    pandas DataFrames found in SSL-FL checkpoints.

    Uses a custom pickle module with an ``Unpickler`` that intercepts
    ``new_block`` calls and fixes the ``slice`` -> ``BlockPlacement``
    type mismatch.

    Args:
        path: Path to the ``.pth`` checkpoint file.
        map_location: Device mapping (default ``"cpu"``).

    Returns:
        The checkpoint dictionary.
    """
    try:
        # Try standard load first (works for our own checkpoints)
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # SSL-FL checkpoint with stale pandas — use custom unpickler
        print("[safe_torch_load] Retrying with pandas-compat unpickler...")
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
            pickle_module=_compat_pickle,
        )
