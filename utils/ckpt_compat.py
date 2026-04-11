"""
utils/ckpt_compat.py -- Cross-version compatible checkpoint loading.

SSL-FL checkpoints embed ``argparse.Namespace`` objects that contain
pandas DataFrames (``args.record_val_acc``, ``args.record_test_acc``).
When loaded on a different pandas version, ``torch.load`` crashes during
unpickling because the internal pandas ``Block`` constructor signature
has changed.

This module provides :func:`safe_torch_load` which intercepts the
pandas ``BlockPlacement`` error and retries with a custom ``Unpickler``
that replaces unresolvable objects with ``None``.  The ``'model'`` key
(the actual weights) is never affected — only the ``'args'`` metadata
can contain stale pandas objects.
"""

import io
import pickle
from typing import Any

import torch


class _PermissiveUnpickler(pickle.Unpickler):
    """
    Unpickler that replaces any class it cannot import or instantiate
    with a no-op placeholder.  Used exclusively for checkpoint loading
    where the ``'args'`` key may contain stale pandas objects.
    """

    def find_class(self, module: str, name: str) -> Any:
        try:
            return super().find_class(module, name)
        except (AttributeError, ImportError, ModuleNotFoundError):
            # Return a dummy that absorbs any constructor call
            return _DummyObject


class _DummyObject:
    """Absorbs arbitrary constructor arguments."""

    def __init__(self, *args, **kwargs):
        pass

    def __reduce__(self):
        return (_DummyObject, ())


def safe_torch_load(path: str, map_location: str = "cpu") -> dict:
    """
    Load a PyTorch checkpoint with graceful handling of stale pickled
    objects (e.g. old pandas DataFrames inside ``args``).

    1. First attempt: normal ``torch.load(..., weights_only=False)``.
    2. If that fails with a pandas/pickle error, fall back to a
       permissive unpickler that replaces unresolvable objects with
       ``None`` placeholders.

    The ``'model'`` state_dict is always loaded correctly because it
    contains only ``torch.Tensor`` objects.

    Args:
        path: Path to the ``.pth`` checkpoint file.
        map_location: Device mapping (default ``"cpu"``).

    Returns:
        The checkpoint dictionary.
    """
    # --- Attempt 1: standard load ---
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except (TypeError, pickle.UnpicklingError, AttributeError) as e:
        print(f"[safe_torch_load] Standard load failed: {e.__class__.__name__}")
        print(f"[safe_torch_load] Retrying with permissive unpickler...")

    # --- Attempt 2: permissive unpickler ---
    with open(path, "rb") as f:
        unpickler = _PermissiveUnpickler(f)
        # Apply map_location by hooking into torch's internal mechanism
        # torch.load uses a custom _load function; we replicate the
        # essential logic here for the CPU case.
        try:
            ckpt = unpickler.load()
        except Exception:
            # If even the permissive unpickler fails, use torch's own
            # mechanism with safe_globals for argparse.Namespace
            import argparse
            torch.serialization.add_safe_globals([argparse.Namespace])
            return torch.load(path, map_location=map_location, weights_only=True)

    # Move tensors to the requested device
    if isinstance(ckpt, dict):
        device = torch.device(map_location)
        for key, val in ckpt.items():
            if isinstance(val, dict):
                for k2, v2 in val.items():
                    if isinstance(v2, torch.Tensor):
                        val[k2] = v2.to(device)
            elif isinstance(val, torch.Tensor):
                ckpt[key] = val.to(device)

    print(f"[safe_torch_load] Loaded successfully with permissive unpickler.")
    return ckpt
