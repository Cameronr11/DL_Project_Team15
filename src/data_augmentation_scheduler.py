"""
Data Augmentation Scheduler for MRNet Training Pipeline

This module implements a three-phase data augmentation scheduling approach:
- Phase 1 (warm-up): Training on raw data to learn core features
- Phase 2 (explore): Gradually increasing augmentation strength to improve generalization
- Phase 3 (fine-tune): Return to clean data for final convergence

This "augmentation sandwich" approach helps combat overfitting while maintaining
training stability and ultimately leads to better generalization.
"""

from typing import Callable, Union
import logging
from src.data_augmentation import ( SimpleMRIAugmentation, MRIAugmentationPipeline, GaussianBlur, RandomCutout)
import torch



# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NoAug:
    """Simple pass-through transform that returns the input unchanged."""
    def __call__(self, x):
        # Convert to tensor if needed
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float()
        return x


class DataAugmentationScheduler:
    """
    Three-phase curriculum scheduler for MRI data augmentation.
    
    The scheduler provides appropriate transforms based on the current epoch:
    - Phase 1: No augmentation (clean data)
    - Phase 2: Progressive augmentation with increasing strength
    - Phase 3: No augmentation again (clean data)
    
    This approach helps the model first learn basic features, then adapt to 
    variations, and finally fine-tune on clean data.
    """
    def __init__(
        self,
        total_epochs: int,
        phase1: Union[float, int] = 0.2,   # proportion 0-1 OR absolute epoch count
        phase2: Union[float, int] = 0.6,   # proportion 0-1 OR absolute epoch count
        max_rot: float = 8.0,              # max rotation degrees at peak of phase 2
        max_brightness: float = 0.08,      # max brightness factor at peak of phase 2
    ):
        """
        Initialize the data augmentation scheduler.
        
        Args:
            total_epochs: Total number of training epochs
            phase1: Duration of phase 1 (no augmentation)
                   - If < 1, treated as a proportion of total_epochs
                   - If >= 1, treated as absolute epoch count
            phase2: Duration of phase 1 + phase 2 (augmentation phase ends here)
                   - If < 1, treated as a proportion of total_epochs
                   - If >= 1, treated as absolute epoch count
            max_rot: Maximum rotation degrees at peak augmentation
            max_brightness: Maximum brightness adjustment factor at peak
        """
        self.total = total_epochs
        
        # Convert phase lengths to absolute epoch counts if needed
        self.p1_end = int(phase1 * total_epochs) if phase1 < 1 else int(phase1)
        self.p2_end = int(phase2 * total_epochs) if phase2 < 1 else int(phase2)
        
        # Ensure phase2 end is after phase1 end
        self.p2_end = max(self.p2_end, self.p1_end + 1)
        
        # Ensure phase2 end is within total epochs
        self.p2_end = min(self.p2_end, total_epochs)
        
        # Augmentation parameters
        self.max_rot = max_rot
        self.max_brightness = max_brightness
        
        # No augmentation transform
        self.no_aug = NoAug()
        
        # Log configuration
        logger.info(f"Initialized DataAugmentationScheduler with total_epochs={total_epochs}")
        logger.info(f"Phase 1 (no aug): epochs 0-{self.p1_end-1}")
        logger.info(f"Phase 2 (progressive aug): epochs {self.p1_end}-{self.p2_end-1}")
        logger.info(f"Phase 3 (no aug): epochs {self.p2_end}-{total_epochs-1}")
        logger.info(f"Max augmentation at epoch {self.p2_end-1}: rotation={max_rot}°, brightness=±{max_brightness*100}%")

    def _interp(self, epoch: int, start: int, end: int) -> float:
        """
        Calculate interpolation factor (0-1) based on current epoch.
        
        Args:
            epoch: Current epoch
            start: Start epoch (corresponds to factor 0)
            end: End epoch (corresponds to factor 1)
            
        Returns:
            float: Interpolation factor between 0 and 1
        """
        return (epoch - start) / max(1, end - start)

    def get_transform(self, epoch: int) -> Callable:
        """
        Get the appropriate transform for the current epoch.
        
        Args:
            epoch: Current training epoch (0-based)
            
        Returns:
            Callable: Transform to apply to the dataset
        """
        # Phase 1: No augmentation
        if epoch < self.p1_end:
            logger.info(f"Epoch {epoch}: Phase 1 - No augmentation")
            return self.no_aug
        
        # Phase 2: Progressive **strong** augmentation
        elif epoch < self.p2_end:
            alpha = self._interp(epoch, self.p1_end, self.p2_end)   # 0 → 1

            # linearly ramp probability & intensity
            p_now = 0.4 + 0.5 * alpha           # 0.4 → 0.9
            rot   = max(3.0, 15.0 * alpha)      # 3° → 15°
            bri   = max(0.02, 0.15 * alpha)     # 2 % → 15 %

            pipeline = MRIAugmentationPipeline(
                [
                    SimpleMRIAugmentation(
                        p=p_now,
                        rotation_degrees=rot,
                        brightness_factor=bri,
                    ),
                    GaussianBlur(p=0.3),                 # ← extra blur
                    RandomCutout(frac=0.25, p=0.4),      # ← small cut-out
                ]
            )

            # (optional) add attributes for debugging
            pipeline.p = p_now
            pipeline.rotation_degrees = rot
            pipeline.brightness_factor = bri

            logger.info(
                "Epoch %d: Phase 2 – p=%.2f  rot=±%.1f°  bri=±%.1f%%",
                epoch, p_now, rot, bri * 100
            )
            return pipeline
        
        # Phase 3: No augmentation again
        else:
            logger.info(f"Epoch {epoch}: Phase 3 - No augmentation")
            return self.no_aug 