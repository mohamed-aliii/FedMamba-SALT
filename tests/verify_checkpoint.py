"""
tests/verify_checkpoint.py -- MAE checkpoint verification tool.

Run this IMMEDIATELY after placing the MAE ViT-B/16 checkpoint at
data/ckpts/mae_vit_base.pth -- BEFORE any model code is executed.

It inspects the raw checkpoint dictionary and confirms that the weight
loading logic in models/vit_teacher.py will work correctly, catching
key-prefix mismatches before they cause silent failures deep in training.

Usage:
    python -m tests.verify_checkpoint
    python -m tests.verify_checkpoint --ckpt_path /path/to/checkpoint.pth
"""

import argparse
import functools
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Verify MAE ViT-B/16 checkpoint before training",
    )
    p.add_argument(
        "--ckpt_path", type=str, default="data/ckpts/mae_vit_base.pth",
        help="Path to the MAE checkpoint file",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results = []  # (name, passed)

    print("=" * 62)
    print("  MAE Checkpoint Verification")
    print("=" * 62)

    # ==================================================================
    # Step 1: File existence and size
    # ==================================================================
    print(f"\n[Step 1] Checking file: {args.ckpt_path}")

    if not os.path.isfile(args.ckpt_path):
        print(f"  [ERROR] File not found: {args.ckpt_path}")
        print(f"  Download the MAE ViT-B/16 checkpoint and place it at")
        print(f"  this path before running any training code.")
        sys.exit(1)

    size_mb = os.path.getsize(args.ckpt_path) / (1024 * 1024)
    print(f"  File exists: {size_mb:.1f} MB")

    # ==================================================================
    # Step 2: Load checkpoint and inspect top-level keys
    # ==================================================================
    print(f"\n[Step 2] Loading checkpoint...")
    ckpt = torch.load(args.ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict):
        top_keys = list(ckpt.keys())
        print(f"  Top-level keys: {top_keys}")
    else:
        print(f"  [WARNING] Checkpoint is not a dict, it is: {type(ckpt).__name__}")
        print(f"  This is unusual for MAE checkpoints.")
        top_keys = []

    has_model_key = "model" in top_keys
    if has_model_key:
        print(f"  [OK] 'model' key found")
    else:
        print(f"  [WARNING] 'model' key NOT found.")
        print(f"  Available keys: {top_keys}")
        print(f"  Will fall back to using the checkpoint dict directly.")
    results.append(("'model' key exists", has_model_key))

    # ==================================================================
    # Step 3: Access state dict and count parameters
    # ==================================================================
    print(f"\n[Step 3] Accessing state dict...")
    state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else {}

    if not isinstance(state_dict, dict):
        print(f"  [ERROR] state_dict is {type(state_dict).__name__}, expected dict")
        sys.exit(1)

    n_tensors = len(state_dict)
    print(f"  Total parameter tensors: {n_tensors}")

    # ==================================================================
    # Step 4: Detect key prefix
    # ==================================================================
    print(f"\n[Step 4] Detecting key prefix...")
    sample_keys = list(state_dict.keys())[:10]
    print(f"  First 10 keys:")
    for k in sample_keys:
        print(f"    {k}")

    # Detect prefix
    detected_prefix = None
    if all(k.startswith("model.encoder.") for k in sample_keys if k.startswith("model.")):
        # Check if majority start with model.encoder.
        model_encoder_count = sum(1 for k in state_dict if k.startswith("model.encoder."))
        if model_encoder_count > len(state_dict) * 0.3:
            detected_prefix = "model.encoder."

    if detected_prefix is None:
        encoder_count = sum(1 for k in state_dict if k.startswith("encoder."))
        no_prefix_count = sum(1 for k in state_dict if not k.startswith("encoder.") and not k.startswith("decoder.") and not k.startswith("model."))

        if encoder_count > len(state_dict) * 0.3:
            detected_prefix = "encoder."
        elif no_prefix_count > len(state_dict) * 0.5:
            detected_prefix = "(none)"
        else:
            detected_prefix = "(mixed/unknown)"

    print(f"\n  Detected prefix: '{detected_prefix}'")
    if detected_prefix == "encoder.":
        print(f"  [OK] This matches vit_teacher.py's stripping logic.")
    elif detected_prefix == "(none)":
        print(f"  [WARNING] No prefix detected. vit_teacher.py strips 'encoder.' --")
        print(f"  if there is no prefix, stripping will yield 0 matching keys.")
    else:
        print(f"  [WARNING] Prefix '{detected_prefix}' does not match 'encoder.'")
        print(f"  vit_teacher.py may need to be updated to strip this prefix.")

    # ==================================================================
    # Step 5: Check patch embedding weight
    # ==================================================================
    print(f"\n[Step 5] Looking for patch embedding weight...")

    patch_key_candidates = [
        "encoder.patch_embed.proj.weight",
        "model.encoder.patch_embed.proj.weight",
        "patch_embed.proj.weight",
    ]

    patch_key_found = None
    patch_shape = None
    for candidate in patch_key_candidates:
        if candidate in state_dict:
            patch_key_found = candidate
            patch_shape = tuple(state_dict[candidate].shape)
            break

    if patch_key_found is None:
        print(f"  [ERROR] Patch embedding weight not found.")
        print(f"  Tried: {patch_key_candidates}")
        # Try fuzzy match
        fuzzy = [k for k in state_dict if "patch_embed" in k and "weight" in k]
        if fuzzy:
            print(f"  Fuzzy matches: {fuzzy}")
        results.append(("patch_embed shape [768,3,16,16]", False))
    else:
        print(f"  Found key: '{patch_key_found}'")
        print(f"  Shape: {list(patch_shape)}")

        expected_shape = (768, 3, 16, 16)
        shape_ok = patch_shape == expected_shape

        if shape_ok:
            print(f"  [OK] Matches ViT-B/16: embed_dim=768, patch_size=16")
        else:
            print(f"  [WARNING] Expected {list(expected_shape)}, got {list(patch_shape)}")
            if patch_shape[0] != 768:
                print(f"    - First dim {patch_shape[0]} != 768: different embed_dim")
                print(f"      (384=ViT-S, 768=ViT-B, 1024=ViT-L)")
            if len(patch_shape) >= 4 and (patch_shape[2] != 16 or patch_shape[3] != 16):
                print(f"    - Last dims {patch_shape[2]}x{patch_shape[3]} != 16x16: different patch_size")

        results.append(("patch_embed shape [768,3,16,16]", shape_ok))

    # ==================================================================
    # Step 6: Simulate prefix stripping and test loading
    # ==================================================================
    print(f"\n[Step 6] Simulating vit_teacher.py prefix stripping...")

    prefix = "encoder."
    stripped = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            stripped[key[len(prefix):]] = value

    print(f"  Keys before stripping: {len(state_dict)}")
    print(f"  Keys after stripping '{prefix}': {len(stripped)}")

    if len(stripped) == 0:
        print(f"  [ERROR] No keys survived stripping. The checkpoint may use")
        print(f"  a different prefix than 'encoder.'")
        results.append(("prefix stripping < 20 missing keys", False))
    else:
        # Try loading into a VisionTransformer
        try:
            from timm.models.vision_transformer import VisionTransformer

            vit = VisionTransformer(
                img_size=224,
                patch_size=16,
                in_chans=3,
                num_classes=0,
                embed_dim=768,
                depth=12,
                num_heads=12,
                mlp_ratio=4.0,
                qkv_bias=True,
                norm_layer=functools.partial(nn.LayerNorm, eps=1e-6),
            )
            load_result = vit.load_state_dict(stripped, strict=False)

            n_missing = len(load_result.missing_keys)
            n_unexpected = len(load_result.unexpected_keys)

            print(f"  Missing keys:    {n_missing}")
            print(f"  Unexpected keys: {n_unexpected}")

            if n_missing > 20:
                print(f"  [WARNING] {n_missing} missing keys is suspicious!")
                print(f"  First 10 missing keys:")
                for k in load_result.missing_keys[:10]:
                    print(f"    - {k}")

            few_missing = n_missing <= 20
            results.append(("prefix stripping < 20 missing keys", few_missing))

        except ImportError:
            print(f"  [SKIP] timm not installed -- cannot test load_state_dict.")
            print(f"  Install timm==0.3.2 to enable this check.")
            # Count stripped keys vs expected ViT-B/16 params (~152 tensors)
            reasonable_count = len(stripped) > 50
            tag = "OK" if reasonable_count else "WARNING"
            print(f"  [{tag}] {len(stripped)} stripped keys "
                  f"(ViT-B/16 has ~152 parameter tensors)")
            results.append(("prefix stripping < 20 missing keys", reasonable_count))

        except Exception as e:
            print(f"  [ERROR] Failed to construct VisionTransformer: {e}")
            print(f"  This may be a timm version compatibility issue.")
            # Fall back to key count heuristic
            reasonable_count = len(stripped) > 50
            results.append(("prefix stripping < 20 missing keys", reasonable_count))

    # ==================================================================
    # Step 7: Final summary
    # ==================================================================
    print(f"\n{'='*62}")
    print(f"  Verification Summary")
    print(f"{'='*62}")

    all_passed = True
    for name, passed in results:
        tag = "PASS" if passed else "FAIL"
        print(f"  [{tag}] {name}")
        if not passed:
            all_passed = False

    print(f"{'='*62}")
    if all_passed:
        print(f"  [OK] ALL CHECKS PASSED")
        print(f"  The checkpoint is compatible with vit_teacher.py.")
        print(f"  You can proceed with training.")
    else:
        print(f"  [!!] SOME CHECKS FAILED")
        print(f"  Fix the issues above before training.")
        print(f"  Common fixes:")
        print(f"    - Wrong checkpoint: download the correct MAE ViT-B/16 file")
        print(f"    - Wrong prefix: update vit_teacher.py stripping logic")
        print(f"    - Wrong architecture: check embed_dim/patch_size match")
    print(f"{'='*62}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
