"""
Metric Tracker for MRNet

This module provides a class to track, store, and retrieve various metrics
during model training and testing for the MRNet knee MRI classification task.
"""

import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, classification_report


class MetricTracker:
    """
    Track, store and visualize metrics for model training and evaluation.
    
    This class provides functionality to:
    - Record various metrics during training (loss, accuracy, AUC, etc.)
    - Save metrics to disk for later analysis
    - Generate visualizations of training progress and model performance
    - Compare performance across different models or configurations
    """
    
    def __init__(self, model_name, task, view, config=None, output_dir=None):
        """
        Initialize the metric tracker.
        
        Args:
            model_name (str): Name of the model (e.g., "resnet18_abnormal_axial")
            task (str): Classification task ('abnormal', 'acl', 'meniscus')
            view (str): MRI view ('axial', 'coronal', 'sagittal')
            config (dict, optional): Configuration parameters used for training
            output_dir (str, optional): Directory to save metrics and visualizations
        """
        self.model_name = model_name
        self.task = task
        self.view = view
        self.config = config or {}
        
        # Create a timestamp for this tracking session
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize metrics storage
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
        self.test_metrics = {}
        self.per_epoch_metrics = []
        
        # Set output directory
        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = os.path.join('results', f"{self.model_name}_{self.timestamp}")
        
        # Create the output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save configuration
        if config:
            self._save_config()
    
    def _save_config(self):
        """Save the configuration to a JSON file."""
        config_path = os.path.join(self.output_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def update_train(self, metrics, step=None):
        """
        Update training metrics.
        
        Args:
            metrics (dict): Dictionary of metric names and values
            step (int, optional): Training step (batch number)
        """
        for name, value in metrics.items():
            self.train_metrics[name].append(value)
        
        # Save step if provided
        if step is not None:
            self.train_metrics['step'].append(step)
    
    def update_val(self, metrics, epoch=None):
        """
        Update validation metrics.
        
        Args:
            metrics (dict): Dictionary of metric names and values
            epoch (int, optional): Current epoch
        """
        for name, value in metrics.items():
            self.val_metrics[name].append(value)
        
        # Save epoch if provided
        if epoch is not None:
            self.val_metrics['epoch'].append(epoch)
            
        # Add to per-epoch metrics for easier analysis
        if epoch is not None:
            epoch_data = {'epoch': epoch, **metrics}
            self.per_epoch_metrics.append(epoch_data)
    
    def update_test(self, metrics):
        """
        Update test metrics.
        
        Args:
            metrics (dict): Dictionary of metric names and values
        """
        self.test_metrics.update(metrics)
    
    def save_metrics(self):
        """Save all metrics to disk."""
        # Save training metrics
        if self.train_metrics:
            # Check if metrics have different lengths
            max_length = max(len(values) for values in self.train_metrics.values())
            
            # Pad arrays to ensure same length
            padded_metrics = {}
            for metric, values in self.train_metrics.items():
                if len(values) < max_length:
                    # Pad with NaN values
                    padded_values = values + [float('nan')] * (max_length - len(values))
                    padded_metrics[metric] = padded_values
                else:
                    padded_metrics[metric] = values
            
            train_df = pd.DataFrame(padded_metrics)
            train_path = os.path.join(self.output_dir, 'train_metrics.csv')
            train_df.to_csv(train_path, index=False)
        
        # Save validation metrics
        if self.val_metrics:
            # Check if metrics have different lengths
            max_length = max(len(values) for values in self.val_metrics.values())
            
            # Pad arrays to ensure same length
            padded_metrics = {}
            for metric, values in self.val_metrics.items():
                if len(values) < max_length:
                    # Pad with NaN values
                    padded_values = values + [float('nan')] * (max_length - len(values))
                    padded_metrics[metric] = padded_values
                else:
                    padded_metrics[metric] = values
            
            val_df = pd.DataFrame(padded_metrics)
            val_path = os.path.join(self.output_dir, 'val_metrics.csv')
            val_df.to_csv(val_path, index=False)
        
        # Save per-epoch metrics
        if self.per_epoch_metrics:
            epoch_df = pd.DataFrame(self.per_epoch_metrics)
            epoch_path = os.path.join(self.output_dir, 'epoch_metrics.csv')
            epoch_df.to_csv(epoch_path, index=False)
        
        # Save test metrics
        if self.test_metrics:
            test_path = os.path.join(self.output_dir, 'test_metrics.json')
            with open(test_path, 'w') as f:
                json.dump(self.test_metrics, f, indent=4)
    
    def plot_metrics(self, metrics=None, save=True):
        """
        Plot training and validation metrics.
        
        Args:
            metrics (list, optional): List of metric names to plot. If None, plot all available metrics.
            save (bool): Whether to save the plots to disk
            
        Returns:
            List of matplotlib figure objects
        """
        figures = []
        
        # Determine which metrics to plot
        if metrics is None:
            # Plot all metrics that are in both train and val
            metrics = []
            for metric in self.train_metrics.keys():
                if metric in self.val_metrics and metric != 'step' and metric != 'epoch':
                    metrics.append(metric)
                    
            # Add any val-only metrics
            for metric in self.val_metrics.keys():
                if metric not in metrics and metric != 'epoch':
                    metrics.append(metric)
        
        # Create a plot for each metric
        for metric in metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot training metric if available
            if metric in self.train_metrics and len(self.train_metrics[metric]) > 0:
                if 'step' in self.train_metrics and len(self.train_metrics['step']) == len(self.train_metrics[metric]):
                    ax.plot(self.train_metrics['step'], self.train_metrics[metric], 'b-', label=f'Train {metric}')
                else:
                    ax.plot(self.train_metrics[metric], 'b-', label=f'Train {metric}')
            
            # Plot validation metric if available
            if metric in self.val_metrics and len(self.val_metrics[metric]) > 0:
                if 'epoch' in self.val_metrics and len(self.val_metrics['epoch']) == len(self.val_metrics[metric]):
                    # For val metrics, plot against epochs with different x-scale
                    ax2 = ax.twiny()
                    ax2.plot(self.val_metrics['epoch'], self.val_metrics[metric], 'r-', label=f'Val {metric}')
                    ax2.set_xlabel('Epoch')
                    ax2.legend(loc='upper right')
                else:
                    ax.plot(self.val_metrics[metric], 'r-', label=f'Val {metric}')
            
            ax.set_xlabel('Step')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} during training')
            ax.legend(loc='upper left')
            ax.grid(True)
            
            figures.append(fig)
            
            if save:
                plot_path = os.path.join(self.output_dir, f'{metric}_plot.png')
                fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        return figures
    
    def plot_confusion_matrix(self, y_true, y_pred, save=True):
        """
        Plot confusion matrix.
        
        Args:
            y_true (array-like): Ground truth labels
            y_pred (array-like): Predicted labels (after thresholding)
            save (bool): Whether to save the plot to disk
            
        Returns:
            matplotlib figure object
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {self.task} - {self.view}')
        
        if save:
            plot_path = os.path.join(self.output_dir, 'confusion_matrix.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            
        return plt.gcf()
    
    def plot_roc_curve(self, y_true, y_score, save=True):
        """
        Plot ROC curve.
        
        Args:
            y_true (array-like): Ground truth labels
            y_score (array-like): Raw prediction scores (before thresholding)
            save (bool): Whether to save the plot to disk
            
        Returns:
            matplotlib figure object
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = np.trapz(tpr, fpr)  # Area under ROC curve
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.task} - {self.view}')
        plt.legend(loc='lower right')
        
        if save:
            plot_path = os.path.join(self.output_dir, 'roc_curve.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            
        return plt.gcf()
    
    def plot_precision_recall_curve(self, y_true, y_score, save=True):
        """
        Plot precision-recall curve.
        
        Args:
            y_true (array-like): Ground truth labels
            y_score (array-like): Raw prediction scores (before thresholding)
            save (bool): Whether to save the plot to disk
            
        Returns:
            matplotlib figure object
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {self.task} - {self.view}')
        
        if save:
            plot_path = os.path.join(self.output_dir, 'precision_recall.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            
        return plt.gcf()
    
    def generate_summary(self, y_true=None, y_score=None, y_pred=None, save=True):
        """
        Generate a comprehensive summary of model performance.
        
        Args:
            y_true (array-like, optional): Ground truth labels
            y_score (array-like, optional): Raw prediction scores
            y_pred (array-like, optional): Binary predictions
            save (bool): Whether to save the summary to disk
            
        Returns:
            dict: Summary metrics
        """
        summary = {
            'model': self.model_name,
            'task': self.task,
            'view': self.view,
            'timestamp': self.timestamp
        }
        
        # Add final validation metrics
        for metric, values in self.val_metrics.items():
            if metric != 'epoch' and values:
                summary[f'final_val_{metric}'] = values[-1]
        
        # Add test metrics
        summary.update(self.test_metrics)
        
        # Add classification report if we have the necessary data
        if y_true is not None and y_pred is not None:
            report = classification_report(y_true, y_pred, output_dict=True)
            summary['classification_report'] = report
        
        if save:
            summary_path = os.path.join(self.output_dir, 'model_summary.json')
            with open(summary_path, 'w') as f:
                # Handle converting numpy types to Python native types
                json_str = json.dumps(summary, indent=4, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
                f.write(json_str)
        
        return summary
    
    @staticmethod
    def load_from_directory(directory):
        """
        Load a MetricTracker from a saved directory.
        
        Args:
            directory (str): Path to the directory containing saved metrics
            
        Returns:
            MetricTracker: Loaded metric tracker instance
        """
        # Try to load config
        config_path = os.path.join(directory, 'config.json')
        config = None
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        
        # Extract model_name, task, view from directory name or config
        if config and 'model_name' in config and 'task' in config and 'view' in config:
            model_name = config['model_name']
            task = config['task']
            view = config['view']
        else:
            # Try to extract from directory name
            dir_name = os.path.basename(directory)
            parts = dir_name.split('_')
            if len(parts) >= 3:
                model_name = parts[0]
                task = parts[1]
                view = parts[2]
            else:
                model_name = dir_name
                task = 'unknown'
                view = 'unknown'
        
        # Create new tracker
        tracker = MetricTracker(model_name, task, view, config, output_dir=directory)
        
        # Load training metrics
        train_path = os.path.join(directory, 'train_metrics.csv')
        if os.path.exists(train_path):
            train_df = pd.read_csv(train_path)
            for col in train_df.columns:
                tracker.train_metrics[col] = train_df[col].tolist()
        
        # Load validation metrics
        val_path = os.path.join(directory, 'val_metrics.csv')
        if os.path.exists(val_path):
            val_df = pd.read_csv(val_path)
            for col in val_df.columns:
                tracker.val_metrics[col] = val_df[col].tolist()
        
        # Load per-epoch metrics
        epoch_path = os.path.join(directory, 'epoch_metrics.csv')
        if os.path.exists(epoch_path):
            epoch_df = pd.read_csv(epoch_path)
            tracker.per_epoch_metrics = epoch_df.to_dict('records')
        
        # Load test metrics
        test_path = os.path.join(directory, 'test_metrics.json')
        if os.path.exists(test_path):
            with open(test_path, 'r') as f:
                tracker.test_metrics = json.load(f)
        
        return tracker 