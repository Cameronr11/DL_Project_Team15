# MRNet: Deep Learning for Knee MRI Analysis

## Project Overview

This project implements deep learning models for automatic diagnosis of knee abnormalities from MRI scans using the MRNet dataset. The system detects knee pathologies (ACL tears, meniscal tears, and general abnormalities) across three anatomical views (axial, coronal, and sagittal).

## Table of Contents

- [Key Features](#key-features)
- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Data](#data)
- [Usage Examples](#usage-examples)
- [Training Pipeline](#training-pipeline)
- [Advanced Features](#advanced-features)
- [Utilities](#utilities)
- [Acknowledgments](#acknowledgments)

## Key Features

- **Multi-view MRI Analysis**: Process knee MRI data from axial, coronal, and sagittal views
- **Advanced Model Architecture**: Both single-view and ensemble models with attention mechanisms
- **Robust Data Processing**: Comprehensive loading, normalization, and augmentation pipeline
- **Transfer Learning**: Leverages pre-trained CNNs (ResNet18/34, DenseNet121, AlexNet)
- **Visualization Tools**: Attention maps and augmentation visualization
- **Hyperparameter Optimization**: Integration with Optuna

## Model Architecture

The project implements two main architectures:

### Single-view MRNetModel
- Feature extraction using pretrained backbones (ResNet18/34, DenseNet121, AlexNet)
- Attention-weighted slice pooling for highlighting relevant MRI slices
- Mean-max pooling strategy for robust feature aggregation
- Configurable backbone freezing/unfreezing for transfer learning

### MRNetEnsemble
- Combines predictions from all three anatomical views
- Independent single-view models for each view
- Joint feature fusion via concatenation for final classification

## Project Structure

```
.
├── src/
│   ├── model/
│   │   ├── MRNetModel.py              # Model architectures (single-view & ensemble)
│   │   ├── train_multi_gpu.py         # Training pipeline (currently single-GPU only)
│   │   ├── train_with_optuna.py       # Hyperparameter optimization
│   │   ├── test.py                    # Model evaluation
│   │   └── test_densenet.py           # Backbone comparison utilities
│   ├── utils/
│   │   ├── metric_tracker.py          # Performance metrics tracking
│   │   ├── create_dashboard.py        # Visualization dashboard
│   │   └── visualization.py           # Visualization utilities
│   ├── data_loader.py                 # MRI data loading pipeline
│   ├── data_augmentation.py           # Data augmentation techniques
│   ├── data_normalization.py          # Data normalization methods
│   ├── data_augmentation_scheduler.py # Dynamic augmentation scheduling
│   ├── visualize_augmentations.py     # Augmentation visualization
│   └── visualize_attention.py         # Attention maps visualization
├── random_utility/                    # Helper scripts
│   ├── create_dataset_split.py        # Dataset splitting utilities
│   └── find_data.py                   # Data discovery tools
└── requirements.txt                   # Project dependencies
```

## Requirements

```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.5
pandas>=1.3.0
scikit-learn>=0.24.2
matplotlib>=3.4.3
seaborn>=0.11.2
opencv-python>=4.5.3
tensorboard>=2.6.0
```

## Data

The project works with the MRNet dataset from Stanford ML Group, containing knee MRI scans with:
- Three anatomical planes: axial, coronal, and sagittal
- Labels for three clinical tasks: ACL tears, meniscal tears, and abnormality detection

The `MRNetDataset` class expects a specific directory structure with preprocessed `.npy` files.

## Usage Examples

### Training a Single-View Model

```python
from src.model.train_multi_gpu import train

args = {
    'task': 'acl',                  # Options: 'acl', 'meniscus', 'abnormal'
    'view': 'sagittal',             # Options: 'axial', 'coronal', 'sagittal'
    'backbone': 'resnet34',         # Options: 'resnet18', 'resnet34', 'densenet121', 'alexnet'
    'use_attention': True,          # Enable attention mechanism
    'batch_size': 16,
    'epochs': 50,
    'lr': 1e-4,
    'patience': 10                  # Early stopping patience
}

train(args)
```

### Hyperparameter Optimization

```python
from src.model.train_with_optuna import run_optuna_study

study = run_optuna_study(
    task='acl',
    n_trials=50,
    study_name='mrnet_optimization'
)
```

### Inference and Evaluation

```python
from src.model.test import evaluate_model

metrics = evaluate_model(
    model_path='./models/best_acl_model.pth',
    task='acl',
    view='ensemble'  # Or 'axial', 'coronal', 'sagittal'
)
```

## Training Pipeline

The training pipeline includes:

1. **Data Processing**: Custom `MRNetDataset` for efficient loading of MRI volumes
2. **Dynamic Augmentation**: Three-phase augmentation scheduling via `DataAugmentationScheduler`
3. **Transfer Learning**: Gradual backbone unfreezing for better feature extraction
4. **Optimization**: AdamW optimizer with differential learning rates for backbone vs. head
5. **Early Stopping**: Dual early stopping based on AUC and loss metrics
6. **Visualization**: TensorBoard integration for performance tracking

## Advanced Features

- **Mixed Precision Training**: Via PyTorch's autocast for faster training
- **Class-Balanced Loss**: Handles imbalanced medical data distribution
- **Attention Visualization**: Interpretability tools for model decision analysis
- **Customizable Backbones**: Support for various CNN architectures
- **Custom Collation**: Handles variable slice counts across MRI volumes

## Utilities

- **Dataset Splitting**: Tools to create train/validation/test splits
- **Performance Visualization**: Plotting utilities for metrics and model attention
- **Augmentation Previews**: Visual inspection of augmentation effects

## Acknowledgments

This project is inspired by the Stanford ML Group's MRNet work:
https://stanfordmlgroup.github.io/projects/mrnet/

