"""
MRI Data Augmentation Module

This module provides classes for augmenting MRI data to improve model generalization.
Augmentations are carefully chosen to be medically appropriate, ensuring they
don't create unrealistic artifacts or distortions that could affect diagnosis.

The module offers multiple augmentation strategies with different intensity levels:
- Simple augmentations (rotation, brightness)
- More advanced augmentations (optional)
- Configurable probability settings

Usage:
    augmenter = SimpleMRIAugmentation(p=0.5)
    augmented_volume = augmenter(mri_volume)
"""

import torch
import numpy as np
import random
import torchvision.transforms.functional as F
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Helper transforms (slice‑wise)
# ─────────────────────────────────────────────────────────────────────────────
class GaussianBlur:
    """Gaussian blur each slice with probability *p*."""

    def __init__(
        self,
        sigma: tuple[float, float] = (0.3, 1.2),
        p: float = 0.3,
        kernel_size: int = 5,
    ):
        self.sigma, self.p, self.k = sigma, p, kernel_size

    def __call__(self, vol: torch.Tensor) -> torch.Tensor:
        # vol shape (S, C, H, W)
        if torch.rand(1) > self.p:
            return vol

        sig = random.uniform(*self.sigma)

        # Prefer torch ≥ 2.1’s built-in blur; fall back to torchvision on older builds
        try:
            import torch.nn.functional as NF
            return NF.gaussian_blur(vol, (self.k, self.k), (sig, sig))
        except AttributeError:
            import torchvision.transforms.functional as TF
            blurred = [TF.gaussian_blur(slice_, self.k, [sig, sig]) for slice_ in vol]
            return torch.stack(blurred)


class RandomCutout:
    """Zero a random rectangle (≈ frac of area) in each slice."""

    def __init__(self, frac: float = 0.25, p: float = 0.4):
        self.frac, self.p = frac, p

    def __call__(self, vol: torch.Tensor) -> torch.Tensor:
        # vol shape (S, C, H, W)
        if torch.rand(1) > self.p:
            return vol
        s, c, h, w = vol.shape
        ch, cw = int(h * self.frac), int(w * self.frac)
        y0 = random.randint(0, h - ch)
        x0 = random.randint(0, w - cw)
        vol[..., y0 : y0 + ch, x0 : x0 + cw] = 0
        return vol


# ─────────────────────────────────────────────────────────────────────────────
# Core simple augmentation
# ─────────────────────────────────────────────────────────────────────────────
class SimpleMRIAugmentation:
    """Small rotations & brightness shifts (medically safe)."""

    def __init__(
        self,
        p: float = 0.5,
        rotation_degrees: float = 5.0,
        brightness_factor: float = 0.05,
        seed: int | None = None,
    ):
        self.p = p
        self.rotation_degrees = rotation_degrees
        self.brightness_factor = brightness_factor

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)

        logger.info(
            "Initialized SimpleMRIAugmentation with p=%s, rotation=±%s°, brightness=±%s%%",
            p,
            rotation_degrees,
            brightness_factor * 100,
        )

    # ---------------------------------------------------------------------
    def __call__(self, volume):
        if not isinstance(volume, torch.Tensor):
            volume = torch.from_numpy(volume).float()

        original_shape = volume.shape
        original_dtype = volume.dtype

        # Ensure (S,C,H,W)
        if len(original_shape) == 3:
            volume = volume.unsqueeze(1)

        if torch.rand(1) < self.p:
            volume = self._apply_rotation(volume)
        if torch.rand(1) < self.p:
            volume = self._apply_brightness(volume)

        if len(original_shape) == 3:
            volume = volume.squeeze(1)
        return volume.to(original_dtype)

    # ------------------------------------------------------------------
    def _apply_rotation(self, volume):
        angle = torch.rand(1) * 2 * self.rotation_degrees - self.rotation_degrees
        slices = [F.rotate(volume[i], angle.item()) for i in range(volume.shape[0])]
        return torch.stack(slices)

    def _apply_brightness(self, volume):
        factor = 1.0 + (torch.rand(1) * 2 * self.brightness_factor - self.brightness_factor)
        return volume * factor


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline container
# ─────────────────────────────────────────────────────────────────────────────
class MRIAugmentationPipeline:
    """Sequentially apply a list of augmentations."""

    def __init__(self, augmentations: list | None = None):
        self.augmentations = augmentations or [SimpleMRIAugmentation()]
        logger.info("MRIAugmentationPipeline with %d steps", len(self.augmentations))

    def __call__(self, volume):
        if not isinstance(volume, torch.Tensor):
            volume = torch.from_numpy(volume).float()
        for aug in self.augmentations:
            volume = aug(volume)
        return volume


# ─────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ─────────────────────────────────────────────────────────────────────────────
IMAGENET_ROT = 10.0  # degrees
IMAGENET_BRI = 0.12  # fraction


def get_mri_augmentation(mode: str = "simple", p: float = 0.5):
    mode = mode.lower()
    if mode == "none":
        logger.info("Using no augmentation")
        return lambda x: torch.from_numpy(x).float() if not isinstance(x, torch.Tensor) else x

    if mode == "simple":
        logger.info("Using simple augmentation, p=%s", p)
        return SimpleMRIAugmentation(p=p)

    if mode == "standard":
        logger.info("Using standard augmentation, p=%s", p)
        return MRIAugmentationPipeline(
            [
                SimpleMRIAugmentation(p=p, rotation_degrees=IMAGENET_ROT, brightness_factor=IMAGENET_BRI),
                GaussianBlur(p=0.3),
                RandomCutout(frac=0.25, p=0.4),
            ]
        )

    logger.warning("Unknown augmentation mode '%s', defaulting to 'simple'", mode)
    return SimpleMRIAugmentation(p=p)

