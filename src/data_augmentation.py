import torch
import numpy as np
import torchvision.transforms.functional as F

class SimpleMRIAugmentation:
    def __init__(self, p=0.5):
        """
        Simple, safe augmentations for MRI data.
        Args:
            p (float): Probability of applying each augmentation
        """
        self.p = p
        # Very conservative ranges
        self.rotation_degrees = 5.0  # Small rotation only
        self.brightness_factor = 0.05  # Small intensity adjustment (±5%)

    def __call__(self, volume):
        """
        Args:
            volume (torch.Tensor): Shape (num_slices, channels, H, W)
        Returns:
            torch.Tensor: Augmented volume
        """
        if not isinstance(volume, torch.Tensor):
            volume = torch.from_numpy(volume)

        # Only apply augmentations during training with probability p
        if torch.rand(1) < self.p:
            # Small random rotation
            angle = torch.rand(1) * 2 * self.rotation_degrees - self.rotation_degrees
            volume = torch.stack([
                F.rotate(slice, angle.item()) 
                for slice in volume
            ])

        if torch.rand(1) < self.p:
            # Small intensity adjustment
            factor = 1.0 + (torch.rand(1) * 2 * self.brightness_factor - self.brightness_factor)
            volume = volume * factor

        return volume
