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
    Single-view MRNet with mean-max pooling and an optional slice-attention gate.
    Backbone convs stay frozen at init unless unfreeze() is called later.
    """
    def __init__(self, backbone: str = "resnet18", train_backbone: bool = False):
        super().__init__()
        self.backbone_type = backbone.lower()

        # ---------- 1.  Feature extractor ----------
        if self.backbone_type == "resnet18":
            net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.feature_dim = 512
            self.feature_extractor = nn.Sequential(*list(net.children())[:-1])  # (B,512,1,1)
        elif self.backbone_type == "resnet34":
            net = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            self.feature_dim = 512                       # same dimensionality as resnet18
            self.feature_extractor = nn.Sequential(*list(net.children())[:-1])
        elif self.backbone_type == "densenet121":
            net = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            self.feature_dim = net.classifier.in_features           # 1024
            self.feature_extractor = nn.Sequential(*list(net.children())[:-1])  # (B,1024,7,7)
        elif self.backbone_type == "alexnet":
            net = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
            self.feature_dim = 256
            self.feature_extractor = net.features                   # (B,256,6,6)
        else:
            raise ValueError(f"Unsupported backbone {backbone}")

        # freeze convs by default
        if not train_backbone:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False

        # project DenseNet/AlexNet to 1×1 like ResNet for pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # ---------- 2.  Attention gate across slices ----------
        self.slice_attn = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1, bias=False)      # scalar weight per slice
        )

        # ---------- 3.  Classifier ----------
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.feature_dim * 2),  # for concat of mean & max
            nn.Dropout(0.5),
            nn.Linear(self.feature_dim * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    @torch.no_grad()
    def unfreeze(self, pct=1.0):
        """Gradually unfreeze a fraction pct ∈ (0,1] of backbone layers."""
        total = len(list(self.feature_extractor.parameters()))
        
        # Special case for full unfreezing to ensure all parameters are unfrozen
        if pct >= 1.0:
            for p in self.feature_extractor.parameters():
                p.requires_grad = True
            return
            
        # Handle partial unfreezing
        for i, p in enumerate(self.feature_extractor.parameters()):
            if i / total >= 1 - pct:
                p.requires_grad = True

    def forward(self, x):
        # x shape  (B, S, C, H, W)
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)

        feats = self.feature_extractor(x)          # (B·S, C', h?, w?)
        feats = self.global_pool(feats).flatten(1) # (B·S, C')
        feats = feats.view(B, S, -1)               # (B, S, C')

        # attention weights → (B,S,1) then softmax
        attn = self.slice_attn(feats).softmax(dim=1)   # (B,S,1)

        mean_pool = (feats * attn).sum(dim=1)          # (B, C')
        max_pool  = feats.max(dim=1).values            # (B, C')
        pooled    = torch.cat([mean_pool, max_pool], dim=1)

        return self.classifier(pooled)
