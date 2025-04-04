"""
MRNet Model Implementation

This module provides the neural network model architecture used in the MRNet project
for knee MRI abnormality detection. It implements both:
- A single-view model (MRNetModel) that processes MRI data from one anatomical view
- An ensemble model (MRNetEnsemble) that can combine predictions from multiple views

The architecture is based on the approach described in the Stanford ML Group's MRNet paper:
https://stanfordmlgroup.github.io/projects/mrnet/
"""

import torch
import torch.nn as nn
import torchvision.models as models
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MRNetModel(nn.Module):
    """
    MRNet model architecture for single-view MRI classification.
    
    Uses a pretrained CNN backbone (default: AlexNet) to extract features from each MRI slice,
    then applies max pooling across slices to get the most important features from the volume.
    Finally, a classifier MLP makes the final abnormality prediction.
    """
    
    def __init__(self, backbone='alexnet'):
        """
        Initialize the MRNet model.
        
        Args:
            backbone (str): The pretrained model to use as feature extractor.
                            Options: 'alexnet', 'resnet18', 'resnet34', 'densenet121'
        """
        super(MRNetModel, self).__init__()
        
        # Set up the feature extractor backbone
        if backbone == 'alexnet':
            self.backbone = models.alexnet(pretrained=True)
            feature_dim = 256 * 6 * 6  # AlexNet's output dimension
            self.feature_extractor = self.backbone.features
        elif backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            feature_dim = 512  # ResNet18's output dimension
            self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=True)
            feature_dim = 512  # ResNet34's output dimension (same as ResNet18)
            self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
        elif backbone == 'densenet121':
            self.backbone = models.densenet121(pretrained=True)
            feature_dim = self.backbone.classifier.in_features  # DenseNet specific
            self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Global average pooling for variable slice count
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # MLP classifier that takes feature vectors and outputs abnormality prediction
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1)
        )
        
        logger.info(f"Initialized MRNetModel with {backbone} backbone")
        
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape [batch_size, num_slices, channels, height, width]
                For MRNet this is typically [batch_size, variable_slices, 3, 224, 224]
        
        Returns:
            Tensor of shape [batch_size, 1] with the abnormality prediction
        """
        # Extract dimensions from input
        batch_size, num_slices, channels, height, width = x.shape
        
        # Reshape to process all slices as a batch
        x = x.view(batch_size * num_slices, channels, height, width)
        
        # Extract features from all slices
        features = self.feature_extractor(x)
        
        # Get the shape of the feature tensor for proper reshaping
        feature_shape = features.shape
        
        # Get the total elements per slice to avoid shape mismatches
        elements_per_slice = features.size(1) * features.size(2) * features.size(3) if len(feature_shape) > 3 else features.size(1)
        total_elements = features.numel()
        elements_per_batch = total_elements // batch_size
        
        # Reshape back to separate batches and slices dynamically based on feature dimensions
        if isinstance(self.backbone, models.AlexNet):
            # For AlexNet, features shape is [batch*slices, 256, 6, 6]
            features = features.view(batch_size, num_slices, 256, 6, 6)
        elif isinstance(self.backbone, models.DenseNet):
            # For DenseNet, the features need to be flattened first due to its structure
            features = self.global_pool(features)  # Apply global pooling to get [batch*slices, features, 1, 1]
            features = features.view(batch_size, num_slices, -1)  # Reshape to [batch, slices, features]
        else:  # For ResNet
            # For ResNet, features shape is [batch*slices, 512, 1, 1]
            features = features.view(batch_size, num_slices, 512, 1, 1)
        
        # Apply max pooling across slices to get the most important features
        if isinstance(self.backbone, models.DenseNet):
            # Already flattened, just max pool across slices
            features = torch.max(features, dim=1)[0]
        else:
            # For other models, max pool then flatten
            features = torch.max(features, dim=1)[0]
            features = features.view(batch_size, -1)
        
        # Apply the classifier to get final prediction
        output = self.classifier(features)
        
        return output
