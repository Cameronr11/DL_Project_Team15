import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import roc_auc_score
import argparse
import time
import matplotlib.pyplot as plt
import pandas as pd

from src.data_loader import MRNetDataset
from src.model.MRNetModel import MRNetModel, MRNetEnsemble

def get_project_root():
    """Returns the absolute path to the project root directory"""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def verify_gpu():
    """Verify GPU availability and print information"""
    print("\n=== GPU Information ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    else:
        print("WARNING: CUDA is not available. Training will proceed on CPU.")
        print("This will be significantly slower than GPU training.")
    print("=====================\n")

def train_model(args):
    # Verify GPU availability
    verify_gpu()
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Get project root
    project_root = get_project_root()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up tensorboard
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'))
    
    # Create datasets
    train_dataset = MRNetDataset(
        root_dir=project_root,
        task=args.task,
        train=True,
        transform=None  # Add transforms if needed
    )
    
    valid_dataset = MRNetDataset(
        root_dir=project_root,
        task=args.task,
        train=False,
        transform=None  # Add transforms if needed
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(valid_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=MRNetDataset.custom_collate,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=MRNetDataset.custom_collate,
        pin_memory=True
    )
    
    # Create model based on training approach
    if args.train_approach == 'per_view':
        # Train a separate model for each view
        models = {}
        optimizers = {}
        
        for view in ['axial', 'coronal', 'sagittal']:
            model = MRNetModel(backbone=args.backbone)
            model = model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            models[view] = model
            optimizers[view] = optimizer
            
        # Only train the specified view if provided
        if args.view:
            if args.view not in models:
                raise ValueError(f"Invalid view: {args.view}. Choose from 'axial', 'coronal', 'sagittal'")
            models = {args.view: models[args.view]}
            optimizers = {args.view: optimizers[args.view]}
    #Ensemble Method is not working        
    elif args.train_approach == 'ensemble':
        # Train ensemble model
        model = MRNetEnsemble(backbone=args.backbone)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        models = {'ensemble': model}
        optimizers = {'ensemble': optimizer}
    else:
        raise ValueError(f"Invalid training approach: {args.train_approach}")
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    best_auc = {model_name: 0.0 for model_name in models}
    
    for epoch in range(args.epochs):
        # Training phase
        for model_name, model in models.items():
            model.train()
            running_loss = 0.0
            train_pred = []
            train_true = []
            
            for i, batch in enumerate(train_loader):
                # Skip batches without required views
                if args.train_approach == 'per_view' and model_name not in batch['available_views']:
                    continue
                
                # Get labels
                labels = batch['label'].to(device)
                # Reshape labels to match output shape [batch_size, 1]
                labels = labels.view(-1, 1)
                
                if args.train_approach == 'per_view':
                    # For per-view training, use just that view
                    if model_name not in batch['available_views']:
                        continue
                    
                    data = batch[model_name].to(device)
                    optimizer = optimizers[model_name]
                    
                    # Forward pass
                    optimizer.zero_grad()
                    outputs = model(data)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                # Ensemble Method is not working    
                elif args.train_approach == 'ensemble':
                    # For ensemble training, pass all views
                    data_dict = {view: batch[view].to(device) for view in batch['available_views']}
                    
                    if not data_dict:  # Skip if no views available
                        continue
                    
                    # Forward pass
                    optimizer = optimizers['ensemble']
                    optimizer.zero_grad()
                    outputs = model(data_dict)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                
                # Update metrics
                running_loss += loss.item()
                train_pred.extend(torch.sigmoid(outputs).cpu().detach().numpy())
                train_true.extend(labels.cpu().numpy())
                
                # Print update
                if i % 10 == 9:  # print every 10 mini-batches
                    print(f'[{epoch + 1}, {i + 1}] {model_name} loss: {running_loss / 10:.3f}')
                    running_loss = 0.0
            
            # Calculate train AUC
            train_auc = roc_auc_score(train_true, train_pred)
            print(f'{model_name} Train AUC: {train_auc:.3f}')
            writer.add_scalar(f'{model_name}/train_auc', train_auc, epoch)
        
        # Validation phase
        with torch.no_grad():
            for model_name, model in models.items():
                model.eval()
                val_loss = 0.0
                val_pred = []
                val_true = []
                
                for batch in valid_loader:
                    # Skip batches without required views
                    if args.train_approach == 'per_view' and model_name not in batch['available_views']:
                        continue
                    
                    # Get labels
                    labels = batch['label'].to(device)
                    # Reshape labels to match output shape [batch_size, 1]
                    labels = labels.view(-1, 1)
                    
                    if args.train_approach == 'per_view':
                        # For per-view validation, use just that view
                        if model_name not in batch['available_views']:
                            continue
                        
                        data = batch[model_name].to(device)
                        outputs = model(data)
                    #Ensemble Method is not working    
                    elif args.train_approach == 'ensemble':
                        # For ensemble validation, pass all views
                        data_dict = {view: batch[view].to(device) for view in batch['available_views']}
                        
                        if not data_dict:  # Skip if no views available
                            continue
                        
                        outputs = model(data_dict)
                    
                    # Calculate loss
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    # Store predictions and true labels
                    val_pred.extend(torch.sigmoid(outputs).cpu().numpy())
                    val_true.extend(labels.cpu().numpy())
                
                # Calculate validation AUC
                if val_true and val_pred:  # Only calculate if we have predictions
                    val_auc = roc_auc_score(val_true, val_pred)
                    print(f'{model_name} Validation AUC: {val_auc:.3f}')
                    writer.add_scalar(f'{model_name}/val_auc', val_auc, epoch)
                    
                    # Save best model
                    if val_auc > best_auc[model_name]:
                        best_auc[model_name] = val_auc
                        model_path = os.path.join(args.output_dir, f'{model_name}_best.pth')
                        torch.save(model.state_dict(), model_path)
                        print(f"Saved best {model_name} model with AUC: {val_auc:.3f}")
        
        # Save checkpoint for each epoch
        for model_name, model in models.items():
            model_path = os.path.join(args.output_dir, f'{model_name}_epoch_{epoch}.pth')
            torch.save(model.state_dict(), model_path)
    
    # Print final results
    print("Training completed!")
    for model_name, auc in best_auc.items():
        print(f"Best {model_name} validation AUC: {auc:.3f}")
    
    # Close tensorboard writer
    writer.close()

def custom_collate(batch):
    """Custom collate function for DataLoader with variable slices"""
    # Find common views available in all samples
    available_in_all = set(['axial', 'coronal', 'sagittal'])
    for sample in batch:
        available_in_sample = set(sample['available_views'])
        available_in_all = available_in_all.intersection(available_in_sample)
    
    # Create batch with only common views and necessary data
    result = {
        'label': torch.stack([sample['label'] for sample in batch]),
        'case_id': [sample['case_id'] for sample in batch],
        'available_views': list(available_in_all)
    }
    
    # Add tensors for available views (even if not common to all)
    all_available_views = set()
    for sample in batch:
        all_available_views.update(sample['available_views'])
    
    for view in all_available_views:
        # Get samples that have this view
        valid_samples = [sample[view] for sample in batch if view in sample['available_views']]
        if valid_samples:
            result[view] = torch.stack(valid_samples)
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MRNet models')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Root directory of the project (default: project root)')
    parser.add_argument('--task', type=str, default='abnormal',
                        choices=['abnormal', 'acl', 'meniscus'],
                        help='Task to train on')
    parser.add_argument('--view', type=str, default='axial',
                        choices=['axial', 'coronal', 'sagittal'],
                        help='MRI view to train on (for single-view training)')
    
    # Model parameters
    parser.add_argument('--backbone', type=str, default='alexnet',
                        choices=['alexnet', 'resnet18'],
                        help='Backbone architecture')
    parser.add_argument('--train_approach', type=str, default='per_view',
                        choices=['per_view', 'ensemble'],
                        help='Training approach')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='model_outputs',
                        help='Directory to save models and logs')
    
    # Hardware parameters
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA training')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    
    args = parser.parse_args()
    
    # Set data directory to project root if not specified
    if args.data_dir is None:
        args.data_dir = get_project_root()
    
    train_model(args)