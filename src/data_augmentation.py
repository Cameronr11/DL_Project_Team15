<<<<<<< HEAD
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


class SimpleMRIAugmentation:
    """
    Applies simple, safe augmentations for MRI data that preserve diagnostic information.
    
    These augmentations are conservative and suitable for medical imaging:
    - Small random rotations (±5 degrees)
    - Subtle brightness/contrast adjustments (±5%)
    """
    
    def __init__(self, p=0.5, rotation_degrees=5.0, brightness_factor=0.05, seed=None):
        """
        Initialize the augmentation parameters.
        
        Args:
            p (float): Probability of applying each augmentation (0-1)
            rotation_degrees (float): Maximum rotation in degrees
            brightness_factor (float): Maximum brightness adjustment factor
            seed (int, optional): Random seed for reproducibility
        """
        self.p = p
        self.rotation_degrees = rotation_degrees
        self.brightness_factor = brightness_factor
        
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        logger.info(f"Initialized SimpleMRIAugmentation with p={p}, "
                   f"rotation_degrees={rotation_degrees}, brightness_factor={brightness_factor}")
    
    def __call__(self, volume):
        """
        Apply augmentations to an MRI volume.
        
        Args:
            volume (numpy.ndarray or torch.Tensor): MRI volume with shape (num_slices, H, W)
                                                   or (num_slices, channels, H, W)
        
        Returns:
            torch.Tensor: Augmented volume with same shape as input
        """
        # Convert to tensor if needed
        if not isinstance(volume, torch.Tensor):
            volume = torch.from_numpy(volume).float()
        
        # Store original shape to handle both single and multi-channel inputs
        original_shape = volume.shape
        original_dtype = volume.dtype
        
        # Reshape to (num_slices, channels, H, W) if needed
        if len(original_shape) == 3:  # (num_slices, H, W)
            volume = volume.unsqueeze(1)  # Add channel dimension
        
        # Apply rotation augmentation with probability p
        if torch.rand(1) < self.p:
            volume = self._apply_rotation(volume)
            
        # Apply brightness augmentation with probability p
        if torch.rand(1) < self.p:
            volume = self._apply_brightness(volume)
        
        # Restore original shape if needed
        if len(original_shape) == 3:
            volume = volume.squeeze(1)
            
        # Return with same dtype as input
        return volume.to(original_dtype)
    
    def _apply_rotation(self, volume):
        """
        Apply a small random rotation to all slices in the volume.
        
        Args:
            volume (torch.Tensor): MRI volume with shape (num_slices, channels, H, W)
            
        Returns:
            torch.Tensor: Rotated volume
        """
        # Generate a random angle within the specified range
        angle = torch.rand(1) * 2 * self.rotation_degrees - self.rotation_degrees
        
        # Apply the same rotation to all slices
        rotated_slices = []
        for i in range(volume.shape[0]):
            # F.rotate expects (C, H, W) tensor
            rotated_slice = F.rotate(volume[i], angle.item())
            rotated_slices.append(rotated_slice)
            
        return torch.stack(rotated_slices)
    
    def _apply_brightness(self, volume):
        """
        Apply a small random brightness adjustment to the volume.
        
        Args:
            volume (torch.Tensor): MRI volume with shape (num_slices, channels, H, W)
            
        Returns:
            torch.Tensor: Brightness-adjusted volume
        """
        # Generate a random brightness factor
        factor = 1.0 + (torch.rand(1) * 2 * self.brightness_factor - self.brightness_factor)
        
        # Apply brightness adjustment to the whole volume
        return volume * factor


class MRIAugmentationPipeline:
    """
    A pipeline that combines multiple MRI augmentation techniques.
    
    This class allows for combining different augmentations in sequence
    with individual probability settings.
    """
    
    def __init__(self, augmentations=None):
        """
        Initialize the augmentation pipeline.
        
        Args:
            augmentations (list, optional): List of augmentation objects to apply
        """
        if augmentations is None:
            # Default to simple augmentations
            self.augmentations = [SimpleMRIAugmentation()]
        else:
            self.augmentations = augmentations
            
        logger.info(f"Initialized MRIAugmentationPipeline with {len(self.augmentations)} augmentations")
    
    def __call__(self, volume):
        """
        Apply the augmentation pipeline to an MRI volume.
        
        Args:
            volume (numpy.ndarray or torch.Tensor): MRI volume
            
        Returns:
            torch.Tensor: Augmented volume
        """
        # Convert to tensor if needed
        if not isinstance(volume, torch.Tensor):
            volume = torch.from_numpy(volume).float()
            
        # Apply each augmentation in sequence
        for aug in self.augmentations:
            volume = aug(volume)
            
        return volume


# Utility function to create standard augmentation pipelines
def get_mri_augmentation(mode='simple', p=0.5):
    """
    Create an MRI augmentation pipeline based on the specified mode.
    
    Args:
        mode (str): Augmentation strategy:
                    - 'none': No augmentation
                    - 'simple': Basic, safe augmentations
                    - 'standard': Balanced set of augmentations
        p (float): Base probability for applying augmentations
        
    Returns:
        callable: An augmentation object that can be applied to MRI volumes
    """
    if mode.lower() == 'none':
        logger.info("Using no augmentation")
        return lambda x: torch.from_numpy(x).float() if not isinstance(x, torch.Tensor) else x
    
    elif mode.lower() == 'simple':
        logger.info(f"Using simple augmentation with p={p}")
        return SimpleMRIAugmentation(p=p)
    
    elif mode.lower() == 'standard':
        logger.info(f"Using standard augmentation pipeline with p={p}")
        return MRIAugmentationPipeline([
            SimpleMRIAugmentation(p=p, rotation_degrees=5.0, brightness_factor=0.05),
            # Additional augmentations could be added here
        ])
    
    else:
        logger.warning(f"Unknown augmentation mode '{mode}', defaulting to 'simple'")
        return SimpleMRIAugmentation(p=p)
=======
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
>>>>>>> e1c80908228f173060b9a8f591340f1f5bd63a3f
