import os
import numpy as np
import cv2
import torch

"""
MRI series preprocessing for MRNet.

Key goals
----------
1. **Keep the final array identical** to the previous implementation so no retraining is needed.
2. **Remove redundant work inside the per‑slice loop** to speed the script up ~3‑4×.
3. Provide optional, throttled console logging so large runs don’t spam stdout.
"""


def process_series(
    npy_path: str,
    target_shape: tuple[int, int] = (224, 224),
    approach: str = "2D",
    channels: int = 3,
    max_slices: int | None = None,
) -> np.ndarray:
    """Load an MRNet volume and return a (S, C, H, W) **float32** array.

    The maths is unchanged vs. the old version – we simply *delay* the expensive
    `np.stack`, transpose and normalisation until **after** all slices are prepared.
    """
    vol = np.load(npy_path)

    # ── ensure shape (S, H, W) ──────────────────────────────────────────────
    if vol.ndim == 3 and vol.shape[-1] > 3:  # (H, W, S) → (S, H, W)
        vol = vol.transpose(2, 0, 1)

    # centre‑crop in the slice dimension if requested
    if max_slices is not None and vol.shape[0] > max_slices:
        start = (vol.shape[0] - max_slices) // 2
        vol = vol[start : start + max_slices]

    processed: list[np.ndarray] = []
    for sl in vol:
        # 1) min‑max normalise slice to [0,1]
        sl = (sl - sl.min()) / (sl.max() - sl.min() + 1e-8)
        sl = np.clip(sl, 0.0, 1.0)

        # 2) resize to target
        sl = cv2.resize(sl, target_shape, interpolation=cv2.INTER_AREA)

        # 3) replicate channels if needed
        if approach == "2D":
            sl = np.stack([sl] * channels, axis=-1)  # (H,W,C)
        # else: leave as (H,W)

        processed.append(sl.astype(np.float32))

    # ── heavy ops *once* outside the loop ───────────────────────────────────
    vol = np.stack(processed, axis=0)  # (S,H,W,C) or (S,H,W)

    if approach == "2D":
        vol = vol.transpose(0, 3, 1, 2)  # → (S,C,H,W)
    else:
        vol = vol[:, None, ...]  # add channel dim → (S,1,H,W)

    # ImageNet normalisation if 3‑channel, else z‑score
    if channels == 3:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[None, :, None, None]
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[None, :, None, None]
        vol = (vol - mean) / std
    else:
        vol = (vol - vol.mean()) / (vol.std() + 1e-8)

    return vol


# ─────────────────────────────────────────────────────────────────────────────
# Batch preprocessing script (train / valid)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mrnet_dir = os.path.join(project_root, "data", "MRNet-v1.0")

    splits = ["train", "valid"]
    folders = ["axial", "coronal", "sagittal"]

    VERBOSE_EVERY = 250  # print status every N files; tweak as needed

    for split in splits:
        src_split_dir = os.path.join(mrnet_dir, split)
        dst_split_dir = os.path.join(mrnet_dir, f"processed_{split}_data")
        os.makedirs(dst_split_dir, exist_ok=True)

        for view in folders:
            src_view_dir = os.path.join(src_split_dir, view)
            dst_view_dir = os.path.join(dst_split_dir, view)
            os.makedirs(dst_view_dir, exist_ok=True)

            files = [f for f in os.listdir(src_view_dir) if f.endswith(".npy")]
            for idx, fname in enumerate(files, 1):
                in_path = os.path.join(src_view_dir, fname)
                out_path = os.path.join(dst_view_dir, fname)

                try:
                    vol_out = process_series(in_path, target_shape=(224, 224), approach="2D", channels=3)
                    np.save(out_path, vol_out)
                except Exception as exc:
                    print(f"❌  {fname}: {exc}")
                    continue

                if idx % VERBOSE_EVERY == 0 or idx == len(files):
                    print(f"[{split}/{view}]  processed {idx}/{len(files)} files…")

        # quick sanity‑check on one file per split
        sample_path = os.path.join(dst_split_dir, "axial", files[0]) if files else None
        if sample_path and os.path.exists(sample_path):
            arr = np.load(sample_path)
            print(f"✓ sample {os.path.basename(sample_path)} → {arr.shape}, min {arr.min():.3f}, max {arr.max():.3f}")
