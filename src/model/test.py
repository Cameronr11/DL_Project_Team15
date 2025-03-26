import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import numpy as np
import os
import gc
from torch.utils.data import DataLoader
from src.data_loader import MRNetDataset
from src.model.MRNetModel import MRNetModel
import json
import argparse
from src.model.train_multi_gpu import get_project_root

def test_model(args, model, device, test_loader, criterion):
    """
    Evaluate model on test set
    """
    model.eval()
    test_loss = 0
    test_pred = []
    test_true = []
    total_samples = 0
    
    print("\nStarting testing process...")
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # Print progress
            if i % 10 == 0:
                print(f"Processing batch {i+1}/{len(test_loader)}")
            
            # Get labels
            labels = batch['label'].to(device)
            labels = labels.view(-1, 1)
            
            if args.train_approach == 'per_view':
                if args.view not in batch['available_views']:
                    continue
                
                data = batch[args.view].to(device)
                outputs = model(data)
            else:  # ensemble
                data_dict = {view: batch[view].to(device) for view in batch['available_views']}
                outputs = model(data_dict)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            # Store predictions and true labels
            test_pred.extend(torch.sigmoid(outputs).cpu().numpy())
            test_true.extend(labels.cpu().numpy())
            total_samples += labels.size(0)
            
            # Free up memory
            del data, outputs, labels
            torch.cuda.empty_cache()
            gc.collect()
    
    # Calculate metrics
    test_auc = roc_auc_score(test_true, test_pred)
    test_pred_binary = (np.array(test_pred) > 0.5).astype(int)
    conf_matrix = confusion_matrix(test_true, test_pred_binary)
    
    # Print results
    print("\nTest Results:")
    print(f"Processed {total_samples} samples")
    print(f"Average Loss: {test_loss / len(test_loader):.4f}")
    print(f"AUC Score: {test_auc:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(classification_report(test_true, test_pred_binary))
    
    return test_auc, test_loss / len(test_loader)

def selective_collate(batch, view=None):
    """
    Custom collation function that only collates the specific view.
    """
    # Basic collation for scalar values
    collated_batch = {
        'case_id': [item['case_id'] for item in batch],
        'label': torch.stack([item['label'] for item in batch]),
    }
    
    # Find all available views across the batch
    all_views = set()
    for item in batch:
        all_views.update(item['available_views'])
    
    # If a specific view is requested, only include that one
    views_to_include = [view] if view else all_views
    
    # Track which views are actually available in this batch
    available_views = []
    
    # Collate each view separately
    for view in views_to_include:
        if view in all_views:
            # Get all tensors for this view
            view_tensors = []
            for item in batch:
                if view in item['available_views']:
                    view_tensors.append(item[view])
            
            if view_tensors:
                try:
                    # Stack them along the batch dimension
                    collated_batch[view] = torch.stack(view_tensors)
                    available_views.append(view)
                    
                    # Only print for first batch
                    if not hasattr(selective_collate, '_first_batch_logged'):
                        print(f"First batch: Collated {view} tensor with shape: {collated_batch[view].shape}")
                except Exception as e:
                    print(f"ERROR stacking {view} tensors: {str(e)}")
    
    # Mark first batch as logged after processing all views
    if not hasattr(selective_collate, '_first_batch_logged'):
        selective_collate._first_batch_logged = True
    
    collated_batch['available_views'] = available_views
    return collated_batch

def main():
    parser = argparse.ArgumentParser(description='Test MRNet models')
    # Add the same arguments as training script
    parser.add_argument('--task', type=str, required=True,
                      choices=['abnormal', 'acl', 'meniscus'])
    parser.add_argument('--view', type=str, required=True,
                      choices=['axial', 'coronal', 'sagittal'])
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to saved model')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--backbone', type=str, default='resnet18',
                      choices=['alexnet', 'resnet18', 'densenet121'],
                      help='Backbone architecture')
    parser.add_argument('--train_approach', type=str, default='per_view',
                      choices=['per_view', 'ensemble'],
                      help='Training approach (per_view or ensemble)')
    parser.add_argument('--max_slices', type=int, default=32,
                      help='Maximum number of slices to use per MRI (should match training)')
    parser.add_argument('--num_workers', type=int, default=1,
                      help='Number of worker processes for data loading')
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n=== Test Configuration ===")
    print(f"Task: {args.task}")
    print(f"View: {args.view}")
    print(f"Training approach: {args.train_approach}")
    print(f"Backbone: {args.backbone}")
    print(f"Max slices: {args.max_slices}")
    print(f"Model path: {args.model_path}")
    print(f"Device: {device}")
    print("=========================\n")
    
    # Load test dataset
    test_dataset = MRNetDataset(
        root_dir=get_project_root(),
        task=args.task,
        split='test',
        transform=None,
        max_slices=args.max_slices
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Create custom collate function that only processes the needed view
    collate_fn = lambda batch: selective_collate(batch, args.view if args.train_approach == 'per_view' else None)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,  # Reduced from 4 to reduce memory usage
        collate_fn=collate_fn,  # Use view-specific collate function
        pin_memory=True
    )
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = MRNetModel(backbone=args.backbone)
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    
    # Free up memory before starting test
    torch.cuda.empty_cache()
    gc.collect()
    
    # Summarize model architecture
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model loaded successfully!")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Run testing
    test_auc, test_loss = test_model(args, model, device, test_loader, criterion)
    
    # Save results
    results_dir = os.path.join(get_project_root(), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    results = {
        'task': args.task,
        'view': args.view,
        'backbone': args.backbone,
        'train_approach': args.train_approach,
        'test_auc': float(test_auc),  # Convert to float for JSON serialization
        'test_loss': float(test_loss)
    }
    
    results_path = os.path.join(results_dir, 
                               f'test_results_{args.task}_{args.view}_{args.backbone}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to {results_path}")

if __name__ == '__main__':
    main()
