import torch
import torch.nn as nn
import torchvision.models as models

class MRNetModel(nn.Module):
    """
    MRNet model architecture as described in the paper:
    https://stanfordmlgroup.github.io/projects/mrnet/
    
    Uses a pretrained AlexNet backbone with a custom classifier.
    """
    def __init__(self, backbone='alexnet'):
        super(MRNetModel, self).__init__()
        
        # Load the pretrained backbone
        if backbone == 'alexnet':
            self.backbone = models.alexnet(pretrained=True)
            feature_dim = 256 * 6 * 6  # AlexNet's output dimension
        elif backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            feature_dim = 512  # ResNet18's output dimension
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove the classification head
        if backbone == 'alexnet':
            self.feature_extractor = self.backbone.features
        else:  # For ResNet, keep everything except the final FC layer
            self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Global average pooling for variable slice count
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # MLP classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1)
        )
        
    def forward(self, x):
        """
        Forward pass through the model
        
        Args:
            x: Input tensor of shape [batch_size, num_slices, channels, height, width]
                For MRNet this is typically [batch_size, variable_slices, 3, 224, 224]
        
        Returns:
            Tensor of shape [batch_size, 1] with the abnormality prediction
        """
        batch_size, num_slices, channels, height, width = x.shape
        
        # Reshape to process all slices as a batch
        x = x.view(batch_size * num_slices, channels, height, width)
        
        # Extract features from all slices
        features = self.feature_extractor(x)
        
        # Reshape back to separate batches and slices
        if isinstance(self.backbone, models.AlexNet):
            # For AlexNet
            features = features.view(batch_size, num_slices, 256, 6, 6)
        else:
            # For ResNet
            features = features.view(batch_size, num_slices, 512, 1, 1)
        
        # Global max pooling across slices
        features = torch.max(features, dim=1)[0]
        
        # Flatten the features
        features = features.view(batch_size, -1)
        
        # Classification head
        output = self.classifier(features)
        
        return output

##Ensemble Method is not working
class MRNetEnsemble(nn.Module):
    """
    MRNet ensemble model that combines predictions from three separate models,
    one for each view (axial, coronal, sagittal).
    """
    def __init__(self, backbone='alexnet'):
        super(MRNetEnsemble, self).__init__()
        
        # Create a separate model for each view
        self.axial_model = MRNetModel(backbone)
        self.coronal_model = MRNetModel(backbone)
        self.sagittal_model = MRNetModel(backbone)
        
        # Logistic regression to combine the three models
        self.combiner = nn.Linear(3, 1)
        
    def forward(self, data_dict):
        """
        Forward pass through the ensemble model
        
        Args:
            data_dict: Dictionary containing tensors for each view
                {'axial': [batch_size, num_slices_axial, 3, 224, 224],
                 'coronal': [batch_size, num_slices_coronal, 3, 224, 224],
                 'sagittal': [batch_size, num_slices_sagittal, 3, 224, 224]}
        
        Returns:
            Tensor of shape [batch_size, 1] with the combined prediction
        """
        outputs = []
        
        if 'axial' in data_dict:
            axial_output = self.axial_model(data_dict['axial'])
            outputs.append(axial_output)
        
        if 'coronal' in data_dict:
            coronal_output = self.coronal_model(data_dict['coronal'])
            outputs.append(coronal_output)
        
        if 'sagittal' in data_dict:
            sagittal_output = self.sagittal_model(data_dict['sagittal'])
            outputs.append(sagittal_output)
        
        # Combine the outputs from all available views
        if len(outputs) > 1:
            combined = torch.cat(outputs, dim=1)
            return self.combiner(combined)
        elif len(outputs) == 1:
            # If only one view is available, use that output directly
            return outputs[0]
        else:
            # If no views are available (should not happen)
            raise ValueError("No views available for prediction")