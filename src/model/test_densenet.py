import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import numpy as np
import os
import gc
from torch.utils.data import DataLoader
from src.data_loader import MRNetDataset
import json
import argparse
from src.model.train_multi_gpu import get_project_root
from src.utils.metric_tracker import MetricTracker
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve


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


def test_densenet(args, model, device, test_loader, criterion):
    """
    Evaluate DenseNet121 model on test set
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
            
            # Get data for the specified view
            data = batch[args.view].to(device)

            # Select a single slice (e.g., the middle slice)
            # Assuming data is of shape [batch_size, num_slices, channels, height, width]
            data = data[:, data.size(1) // 2, :, :, :]  # Select the middle slice

            # Alternatively, you can use max pooling or averaging
            # data = torch.max(data, dim=1)[0]  # Max pooling across slices
            # data = torch.mean(data, dim=1)  # Average pooling across slices

            outputs = model(data)
            
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
    
    # Calculate additional metrics for reporting
    metrics = {
        "test_loss": test_loss / len(test_loader),
        "test_auc": test_auc,
        "test_samples": total_samples
    }
    
    # Add classification report metrics
    report = classification_report(test_true, test_pred_binary, output_dict=True)
    for class_label, class_metrics in report.items():
        if isinstance(class_metrics, dict):
            for metric_name, metric_value in class_metrics.items():
                metrics[f"test_{class_label}_{metric_name}"] = metric_value
    
    return test_auc, test_loss / len(test_loader), test_true, test_pred, test_pred_binary, metrics


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def plot_roc_curve(y_true, y_score, save_path=None):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = np.trapz(tpr, fpr)  # Area under ROC curve
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def plot_precision_recall_curve(y_true, y_score, save_path=None):
    """Plot precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def main():
    parser = argparse.ArgumentParser(description='Test DenseNet121 model')
    # Add the same arguments as training script
    parser.add_argument('--task', type=str, required=True,
                      choices=['abnormal', 'acl', 'meniscus'])
    parser.add_argument('--view', type=str, required=True,
                      choices=['axial', 'coronal', 'sagittal'])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_slices', type=int, default=32,
                      help='Maximum number of slices to use per MRI (should match training)')
    parser.add_argument('--num_workers', type=int, default=1,
                      help='Number of worker processes for data loading')
    parser.add_argument('--output_dir', type=str, default=None,
                      help='Directory to save test results and plots')
    parser.add_argument('--generate_plots', action='store_true',
                      help='Generate visualizations for test results')
    args = parser.parse_args()
    
    # Setup output directory
    if args.output_dir is None:
        # Create default output directory based on model info
        args.output_dir = os.path.join('results', 'densenet_test_results')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n=== Test Configuration ===")
    print(f"Task: {args.task}")
    print(f"View: {args.view}")
    print(f"Max slices: {args.max_slices}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {device}")
    print("=========================")
    
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
    collate_fn = lambda batch: selective_collate(batch, args.view)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,  # Reduced from 4 to reduce memory usage
        collate_fn=collate_fn,  # Use view-specific collate function
        pin_memory=True
    )
    
    # Load DenseNet121 model
    print("Loading DenseNet121 model...")
    model = models.densenet121(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, 1)  # Adjust for binary classification
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
    
    # Create loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Initialize metric tracker for test results
    metric_tracker = MetricTracker(
        model_name='densenet121',
        task=args.task,
        view=args.view,
        config=vars(args),
        output_dir=args.output_dir
    )
    
    # Evaluate model
    test_auc, test_loss, test_true, test_pred, test_pred_binary, metrics = test_densenet(
        args, model, device, test_loader, criterion
    )
    
    # Update and save metrics
    metric_tracker.update_test(metrics)
    metric_tracker.save_metrics()
    
    # Save predictions for later analysis
    np.save(os.path.join(args.output_dir, 'test_true.npy'), np.array(test_true))
    np.save(os.path.join(args.output_dir, 'test_scores.npy'), np.array(test_pred))
    np.save(os.path.join(args.output_dir, 'test_pred_binary.npy'), np.array(test_pred_binary))
    
    # Generate comprehensive model summary
    summary = metric_tracker.generate_summary(
        y_true=test_true, 
        y_score=test_pred, 
        y_pred=test_pred_binary
    )
    
    # Generate visualizations if requested
    if args.generate_plots:
        print("\nGenerating test result visualizations...")
        
        # Confusion matrix
        plot_confusion_matrix(test_true, test_pred_binary)
        plt_path = os.path.join(args.output_dir, 'confusion_matrix.png')
        plt.savefig(plt_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # ROC curve
        plot_roc_curve(test_true, test_pred)
        plt_path = os.path.join(args.output_dir, 'roc_curve.png')
        plt.savefig(plt_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Precision-Recall curve
        plot_precision_recall_curve(test_true, test_pred)
        plt_path = os.path.join(args.output_dir, 'precision_recall_curve.png')
        plt.savefig(plt_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {args.output_dir}")
    
    print("\nTest evaluation completed!")
    print(f"Final Test AUC: {test_auc:.4f}")
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main() 