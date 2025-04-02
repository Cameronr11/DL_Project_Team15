"""
Model Comparison Utility for MRNet

This script provides functionality to compare the performance of multiple trained models
by analyzing their metrics and generating comparative visualizations.

Example usage:
    python src/utils/compare_models.py --results_dir results --output_dir model_comparison
"""

import os
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.utils.metric_tracker import MetricTracker
from src.utils.visualization import (
    compare_metrics,
    compare_roc_curves,
    create_model_performance_dashboard,
    create_model_comparison_table
)


def find_model_results(results_dir):
    """
    Find all model result directories in the specified results directory.
    
    Args:
        results_dir (str): Path to the directory containing model results
        
    Returns:
        list: List of paths to model result directories
    """
    model_dirs = []
    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if os.path.isdir(item_path):
            # Check if this is a model results directory by looking for specific files
            if (os.path.exists(os.path.join(item_path, 'config.json')) or 
                os.path.exists(os.path.join(item_path, 'test_metrics.json')) or
                os.path.exists(os.path.join(item_path, 'model_summary.json'))):
                model_dirs.append(item_path)
    
    return model_dirs


def filter_models_by_criteria(model_dirs, task=None, view=None, backbone=None):
    """
    Filter model directories based on criteria.
    
    Args:
        model_dirs (list): List of model directory paths
        task (str, optional): Task to filter by ('abnormal', 'acl', 'meniscus')
        view (str, optional): View to filter by ('axial', 'coronal', 'sagittal')
        backbone (str, optional): Backbone to filter by ('alexnet', 'resnet18', 'densenet121')
        
    Returns:
        list: Filtered list of model directory paths
    """
    filtered_dirs = []
    
    for model_dir in model_dirs:
        # Try to load config
        config_path = os.path.join(model_dir, 'config.json')
        if not os.path.exists(config_path):
            config_path = os.path.join(model_dir, 'test_config.json')
        
        if not os.path.exists(config_path):
            # Skip if we can't find config information
            continue
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Apply filters
        if task and config.get('task') != task:
            continue
        if view and config.get('view') != view:
            continue
        if backbone and config.get('backbone') != backbone:
            continue
        
        filtered_dirs.append(model_dir)
    
    return filtered_dirs


def create_comparison_plots(model_dirs, output_dir, metrics_to_compare=None):
    """
    Create comparison plots for the specified models.
    
    Args:
        model_dirs (list): List of model directory paths
        output_dir (str): Directory to save comparison plots
        metrics_to_compare (list, optional): List of metrics to compare
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if metrics_to_compare is None:
        metrics_to_compare = ['loss', 'auc']
    
    # Compare training metrics
    for metric in metrics_to_compare:
        try:
            compare_metrics(
                model_dirs, 
                metric, 
                is_train=True,
                save_path=os.path.join(output_dir, f'train_{metric}_comparison.png')
            )
            plt.close()
        except Exception as e:
            print(f"Error comparing training {metric}: {str(e)}")
    
    # Compare validation metrics
    for metric in metrics_to_compare:
        try:
            compare_metrics(
                model_dirs, 
                metric, 
                is_train=False,
                save_path=os.path.join(output_dir, f'val_{metric}_comparison.png')
            )
            plt.close()
        except Exception as e:
            print(f"Error comparing validation {metric}: {str(e)}")
    
    # Compare ROC curves
    try:
        compare_roc_curves(
            model_dirs,
            save_path=os.path.join(output_dir, 'roc_curves_comparison.png')
        )
        plt.close()
    except Exception as e:
        print(f"Error comparing ROC curves: {str(e)}")
    
    # Create performance dashboard
    try:
        create_model_performance_dashboard(
            results_dir=None,
            model_dirs=model_dirs,
            save_path=os.path.join(output_dir, 'model_performance_dashboard.png')
        )
        plt.close()
    except Exception as e:
        print(f"Error creating performance dashboard: {str(e)}")
    
    # Create comparison table
    try:
        df = create_model_comparison_table(
            results_dir=None,
            model_dirs=model_dirs,
            save_path=os.path.join(output_dir, 'model_comparison_table.csv')
        )
        
        # Also save as JSON for easier parsing
        metrics_json = df.to_dict(orient='records')
        with open(os.path.join(output_dir, 'model_comparison.json'), 'w') as f:
            json.dump(metrics_json, f, indent=4)
    except Exception as e:
        print(f"Error creating comparison table: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='Compare multiple trained MRNet models')
    parser.add_argument('--results_dir', type=str, required=True,
                      help='Directory containing model results')
    parser.add_argument('--output_dir', type=str, default='model_comparison',
                      help='Directory to save comparison results')
    parser.add_argument('--task', type=str, choices=['abnormal', 'acl', 'meniscus'],
                      help='Filter models by task')
    parser.add_argument('--view', type=str, choices=['axial', 'coronal', 'sagittal'],
                      help='Filter models by view')
    parser.add_argument('--backbone', type=str, choices=['alexnet', 'resnet18', 'densenet121'],
                      help='Filter models by backbone architecture')
    parser.add_argument('--metrics', type=str, nargs='+', default=['loss', 'auc'],
                      help='Metrics to compare')
    args = parser.parse_args()
    
    # Ensure results directory exists
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory '{args.results_dir}' does not exist")
        return 1
    
    # Find all model result directories
    all_model_dirs = find_model_results(args.results_dir)
    if not all_model_dirs:
        print(f"No model results found in '{args.results_dir}'")
        return 1
    
    print(f"Found {len(all_model_dirs)} model result directories")
    
    # Filter models by criteria
    filtered_dirs = filter_models_by_criteria(
        all_model_dirs,
        task=args.task,
        view=args.view,
        backbone=args.backbone
    )
    
    if not filtered_dirs:
        print("No models match the specified criteria")
        return 1
    
    print(f"Selected {len(filtered_dirs)} models for comparison:")
    for model_dir in filtered_dirs:
        print(f"  - {os.path.basename(model_dir)}")
    
    # Create comparison plots
    create_comparison_plots(filtered_dirs, args.output_dir, args.metrics)
    
    print(f"Comparison results saved to '{args.output_dir}'")
    return 0


if __name__ == "__main__":
    exit(main()) 