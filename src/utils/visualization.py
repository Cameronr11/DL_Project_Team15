"""
Visualization Utilities for MRNet

This module provides functions to visualize:
1. MRI scan data (individual slices and volumes)
2. Model training progress and performance metrics
3. Comparison between different models or configurations
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import glob
from src.utils.metric_tracker import MetricTracker


def visualize_mri_slice(slice_data, title=None, save_path=None):
    """
    Visualize a single MRI slice.
    
    Args:
        slice_data (ndarray): 2D array representing an MRI slice
        title (str, optional): Plot title
        save_path (str, optional): Path to save the figure
        
    Returns:
        matplotlib figure object
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(slice_data, cmap='gray')
    plt.axis('off')
    
    if title:
        plt.title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def visualize_mri_volume(volume_data, num_slices=9, title=None, save_path=None):
    """
    Visualize multiple slices from an MRI volume.
    
    Args:
        volume_data (ndarray): 3D array representing an MRI volume
        num_slices (int): Number of slices to display
        title (str, optional): Plot title
        save_path (str, optional): Path to save the figure
        
    Returns:
        matplotlib figure object
    """
    # Determine grid size based on number of slices
    grid_size = int(np.ceil(np.sqrt(num_slices)))
    
    # Select slices to display
    total_slices = volume_data.shape[0]
    if total_slices < num_slices:
        indices = range(total_slices)
    else:
        step = total_slices // num_slices
        indices = range(0, total_slices, step)[:num_slices]
    
    # Create figure and grid
    fig = plt.figure(figsize=(12, 12))
    gs = GridSpec(grid_size, grid_size, figure=fig)
    
    # Add slices to grid
    for i, idx in enumerate(indices):
        ax = fig.add_subplot(gs[i // grid_size, i % grid_size])
        ax.imshow(volume_data[idx], cmap='gray')
        ax.set_title(f'Slice {idx}')
        ax.axis('off')
    
    # Add overall title
    if title:
        fig.suptitle(title, fontsize=16)
        fig.subplots_adjust(top=0.92)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def compare_metrics(model_dirs, metric_name, is_train=True, save_path=None):
    """
    Compare a specific metric across multiple models.
    
    Args:
        model_dirs (list): List of directories containing model results
        metric_name (str): Name of the metric to compare
        is_train (bool): Whether to use training metrics (True) or validation metrics (False)
        save_path (str, optional): Path to save the figure
        
    Returns:
        matplotlib figure object
    """
    plt.figure(figsize=(12, 8))
    
    # Load data for each model
    for model_dir in model_dirs:
        tracker = MetricTracker.load_from_directory(model_dir)
        model_name = tracker.model_name
        
        # Get the right metrics
        metrics = tracker.train_metrics if is_train else tracker.val_metrics
        x_axis = 'step' if is_train else 'epoch'
        
        if metric_name in metrics and x_axis in metrics:
            plt.plot(metrics[x_axis], metrics[metric_name], label=model_name)
    
    plt.xlabel('Step' if is_train else 'Epoch')
    plt.ylabel(metric_name)
    plt.title(f'Comparison of {metric_name} across models')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def compare_roc_curves(model_dirs, save_path=None):
    """
    Compare ROC curves across multiple models.
    
    Args:
        model_dirs (list): List of directories containing model results
        save_path (str, optional): Path to save the figure
        
    Returns:
        matplotlib figure object
    """
    plt.figure(figsize=(10, 8))
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--')
    
    # Load data for each model
    for model_dir in model_dirs:
        # Attempt to load true labels and scores
        y_true_path = os.path.join(model_dir, 'test_true.npy')
        y_score_path = os.path.join(model_dir, 'test_scores.npy')
        
        if os.path.exists(y_true_path) and os.path.exists(y_score_path):
            y_true = np.load(y_true_path)
            y_score = np.load(y_score_path)
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = np.trapz(tpr, fpr)  # Area under ROC curve
            
            # Plot ROC curve
            model_name = os.path.basename(model_dir).split('_')[0]
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def create_model_performance_dashboard(results_dir, model_dirs=None, save_path=None):
    """
    Create a comprehensive dashboard of model performance.
    
    Args:
        results_dir (str): Directory containing all model results
        model_dirs (list, optional): List of specific model directories to include.
                                    If None, include all directories in results_dir.
        save_path (str, optional): Path to save the figure
        
    Returns:
        matplotlib figure object
    """
    # If model_dirs not provided, use all directories in results_dir
    if model_dirs is None:
        model_dirs = [os.path.join(results_dir, d) for d in os.listdir(results_dir) 
                     if os.path.isdir(os.path.join(results_dir, d))]
    
    # Extract model names and final performance metrics
    models = []
    auc_scores = []
    loss_values = []
    
    for model_dir in model_dirs:
        # Try to load model summary
        summary_path = os.path.join(model_dir, 'model_summary.json')
        if os.path.exists(summary_path):
            import json
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            
            model_name = summary.get('model', os.path.basename(model_dir))
            models.append(model_name)
            
            # Extract key metrics
            if 'test_auc' in summary:
                auc_scores.append(summary['test_auc'])
            elif 'final_val_auc' in summary:
                auc_scores.append(summary['final_val_auc'])
            else:
                auc_scores.append(None)
            
            if 'test_loss' in summary:
                loss_values.append(summary['test_loss'])
            elif 'final_val_loss' in summary:
                loss_values.append(summary['final_val_loss'])
            else:
                loss_values.append(None)
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Bar chart for AUC scores
    ax1 = fig.add_subplot(gs[0, 0])
    if auc_scores and any(score is not None for score in auc_scores):
        bars = ax1.bar(models, auc_scores)
        ax1.set_title('AUC Scores by Model')
        ax1.set_ylim(0, 1)
        ax1.set_ylabel('AUC Score')
        ax1.set_xticklabels(models, rotation=45, ha='right')
        
        # Add value labels
        for bar, score in zip(bars, auc_scores):
            if score is not None:
                ax1.text(bar.get_x() + bar.get_width()/2, score + 0.01,
                        f'{score:.3f}', ha='center', va='bottom')
    
    # Bar chart for loss values
    ax2 = fig.add_subplot(gs[0, 1])
    if loss_values and any(loss is not None for loss in loss_values):
        bars = ax2.bar(models, loss_values)
        ax2.set_title('Loss Values by Model')
        ax2.set_ylabel('Loss')
        ax2.set_xticklabels(models, rotation=45, ha='right')
        
        # Add value labels
        for bar, loss in zip(bars, loss_values):
            if loss is not None:
                ax2.text(bar.get_x() + bar.get_width()/2, loss + 0.01,
                        f'{loss:.3f}', ha='center', va='bottom')
    
    # ROC curves
    ax3 = fig.add_subplot(gs[1, :])
    
    # Plot diagonal line
    ax3.plot([0, 1], [0, 1], 'k--')
    
    # Load data for each model
    for model_dir in model_dirs:
        # Attempt to load true labels and scores
        y_true_path = os.path.join(model_dir, 'test_true.npy')
        y_score_path = os.path.join(model_dir, 'test_scores.npy')
        
        if os.path.exists(y_true_path) and os.path.exists(y_score_path):
            y_true = np.load(y_true_path)
            y_score = np.load(y_score_path)
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = np.trapz(tpr, fpr)  # Area under ROC curve
            
            # Plot ROC curve
            model_name = os.path.basename(model_dir).split('_')[0]
            ax3.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('ROC Curves Comparison')
    ax3.legend(loc='lower right')
    ax3.grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_training_progress_report(model_dir, save_path=None):
    """
    Create a comprehensive report of training progress for a single model.
    
    Args:
        model_dir (str): Directory containing model results
        save_path (str, optional): Path to save the figure
        
    Returns:
        matplotlib figure object
    """
    # Load metrics
    tracker = MetricTracker.load_from_directory(model_dir)
    
    # Create figure
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 2, figure=fig)
    
    # Training and validation loss
    ax1 = fig.add_subplot(gs[0, 0])
    if 'loss' in tracker.train_metrics and 'loss' in tracker.val_metrics:
        # Plot training loss
        if 'step' in tracker.train_metrics:
            ax1.plot(tracker.train_metrics['step'], tracker.train_metrics['loss'], 'b-', label='Train Loss')
        else:
            ax1.plot(tracker.train_metrics['loss'], 'b-', label='Train Loss')
        
        # Plot validation loss on secondary axis
        ax1_2 = ax1.twinx()
        if 'epoch' in tracker.val_metrics:
            ax1_2.plot(tracker.val_metrics['epoch'], tracker.val_metrics['loss'], 'r-', label='Val Loss')
        else:
            ax1_2.plot(tracker.val_metrics['loss'], 'r-', label='Val Loss')
        
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Training Loss', color='b')
        ax1_2.set_ylabel('Validation Loss', color='r')
        ax1.set_title('Loss During Training')
        
        # Add legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Training and validation AUC
    ax2 = fig.add_subplot(gs[0, 1])
    if 'auc' in tracker.train_metrics and 'auc' in tracker.val_metrics:
        # Plot training AUC
        if 'step' in tracker.train_metrics:
            ax2.plot(tracker.train_metrics['step'], tracker.train_metrics['auc'], 'b-', label='Train AUC')
        else:
            ax2.plot(tracker.train_metrics['auc'], 'b-', label='Train AUC')
        
        # Plot validation AUC on secondary axis
        ax2_2 = ax2.twinx()
        if 'epoch' in tracker.val_metrics:
            ax2_2.plot(tracker.val_metrics['epoch'], tracker.val_metrics['auc'], 'r-', label='Val AUC')
        else:
            ax2_2.plot(tracker.val_metrics['auc'], 'r-', label='Val AUC')
        
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Training AUC', color='b')
        ax2_2.set_ylabel('Validation AUC', color='r')
        ax2.set_title('AUC During Training')
        
        # Add legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
    
    # Per-epoch metrics
    ax3 = fig.add_subplot(gs[1, :])
    if tracker.per_epoch_metrics:
        # Convert to DataFrame for easier plotting
        epoch_df = pd.DataFrame(tracker.per_epoch_metrics)
        
        # Plot all available metrics
        metrics_to_plot = [col for col in epoch_df.columns if col != 'epoch']
        for metric in metrics_to_plot:
            ax3.plot(epoch_df['epoch'], epoch_df[metric], 'o-', label=metric)
        
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Metric Value')
        ax3.set_title('Metrics by Epoch')
        ax3.legend()
        ax3.grid(True)
    
    # ROC curve
    ax4 = fig.add_subplot(gs[2, 0])
    
    # Try to load true labels and scores
    y_true_path = os.path.join(model_dir, 'test_true.npy')
    y_score_path = os.path.join(model_dir, 'test_scores.npy')
    
    if os.path.exists(y_true_path) and os.path.exists(y_score_path):
        y_true = np.load(y_true_path)
        y_score = np.load(y_score_path)
        
        # Calculate and plot ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = np.trapz(tpr, fpr)  # Area under ROC curve
        
        ax4.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax4.plot([0, 1], [0, 1], 'k--')
        ax4.set_xlabel('False Positive Rate')
        ax4.set_ylabel('True Positive Rate')
        ax4.set_title('ROC Curve')
        ax4.legend(loc='lower right')
        ax4.grid(True)
    
    # Confusion matrix
    ax5 = fig.add_subplot(gs[2, 1])
    
    if os.path.exists(y_true_path) and os.path.exists(y_score_path):
        # Calculate and plot confusion matrix
        y_pred = (y_score > 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5)
        ax5.set_xlabel('Predicted')
        ax5.set_ylabel('True')
        ax5.set_title('Confusion Matrix')
    
    # Add overall title
    model_name = tracker.model_name
    task = tracker.task
    view = tracker.view
    fig.suptitle(f'Training Progress Report: {model_name} - {task} - {view}', fontsize=16)
    fig.subplots_adjust(top=0.95)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_model_comparison_table(results_dir, save_path=None):
    """
    Create a table comparing key metrics across models.
    
    Args:
        results_dir (str): Directory containing all model results
        save_path (str, optional): Path to save the table as a CSV
        
    Returns:
        pandas.DataFrame: Table of model comparisons
    """
    # Find all model directories
    model_dirs = [os.path.join(results_dir, d) for d in os.listdir(results_dir) 
                 if os.path.isdir(os.path.join(results_dir, d))]
    
    # Extract data for each model
    data = []
    for model_dir in model_dirs:
        # Try to load model summary
        summary_path = os.path.join(model_dir, 'model_summary.json')
        if os.path.exists(summary_path):
            import json
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            
            model_name = summary.get('model', os.path.basename(model_dir))
            task = summary.get('task', 'unknown')
            view = summary.get('view', 'unknown')
            
            model_data = {
                'model': model_name,
                'task': task,
                'view': view,
                'timestamp': summary.get('timestamp', '')
            }
            
            # Add test metrics
            for key, value in summary.items():
                if key.startswith('test_') or key.startswith('final_val_'):
                    model_data[key] = value
            
            # Add configuration parameters
            config_path = os.path.join(model_dir, 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                for key, value in config.items():
                    if key not in model_data:
                        model_data[f'config_{key}'] = value
            
            data.append(model_data)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV if requested
    if save_path and not df.empty:
        df.to_csv(save_path, index=False)
    
    return df 