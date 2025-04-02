"""
Model Performance Dashboard Generator for MRNet

This script creates an HTML dashboard to visualize model performance metrics,
making it easy to compare different models and their training progress.

Example usage:
    python src/utils/create_dashboard.py --results_dir results --output_file dashboard.html
"""

import os
import argparse
import json
import pandas as pd
import numpy as np
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from src.utils.metric_tracker import MetricTracker
from src.utils.visualization import (
    compare_metrics,
    compare_roc_curves,
    create_model_performance_dashboard,
    create_model_comparison_table
)


def fig_to_base64(fig):
    """
    Convert a matplotlib figure to a base64 encoded string for embedding in HTML.
    
    Args:
        fig (matplotlib.figure.Figure): The figure to convert
        
    Returns:
        str: Base64 encoded string
    """
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_str


def generate_model_section(model_dir):
    """
    Generate HTML content for a single model.
    
    Args:
        model_dir (str): Path to the model results directory
        
    Returns:
        str: HTML content for the model section
    """
    try:
        # Load model metrics
        tracker = MetricTracker.load_from_directory(model_dir)
        
        # Load model configuration
        config_path = os.path.join(model_dir, 'config.json')
        if not os.path.exists(config_path):
            config_path = os.path.join(model_dir, 'test_config.json')
        
        config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        
        # Get key model info
        model_name = tracker.model_name
        task = tracker.task
        view = tracker.view
        
        # Generate plots
        # Training progress plot
        train_progress_fig = None
        if tracker.train_metrics and tracker.val_metrics:
            metrics_to_plot = []
            for metric in tracker.train_metrics.keys():
                if metric in tracker.val_metrics and metric not in ['step', 'epoch']:
                    metrics_to_plot.append(metric)
            
            if metrics_to_plot:
                train_progress_fig = plt.figure(figsize=(10, 6))
                for metric in metrics_to_plot:
                    if 'step' in tracker.train_metrics:
                        plt.plot(tracker.train_metrics['step'], tracker.train_metrics[metric], 
                                 'b-', label=f'Train {metric}')
                    else:
                        plt.plot(tracker.train_metrics[metric], 'b-', label=f'Train {metric}')
                    
                    if 'epoch' in tracker.val_metrics:
                        plt2 = plt.twinx()
                        plt2.plot(tracker.val_metrics['epoch'], tracker.val_metrics[metric], 
                                  'r-', label=f'Val {metric}')
                    else:
                        plt.plot(tracker.val_metrics[metric], 'r-', label=f'Val {metric}')
                
                plt.title(f"Training Progress - {model_name}")
                plt.legend()
                plt.grid(True)
        
        # ROC curve
        roc_fig = None
        y_true_path = os.path.join(model_dir, 'test_true.npy')
        y_score_path = os.path.join(model_dir, 'test_scores.npy')
        
        if os.path.exists(y_true_path) and os.path.exists(y_score_path):
            y_true = np.load(y_true_path)
            y_score = np.load(y_score_path)
            
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = np.trapz(tpr, fpr)
            
            roc_fig = plt.figure(figsize=(10, 6))
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend(loc='lower right')
            plt.grid(True)
        
        # Generate HTML content
        html = f"""
        <div class="model-section">
            <h2>{model_name}</h2>
            
            <div class="model-info">
                <table>
                    <tr>
                        <th>Task</th>
                        <td>{task}</td>
                    </tr>
                    <tr>
                        <th>View</th>
                        <td>{view}</td>
                    </tr>
        """
        
        # Add configuration parameters
        for key, value in config.items():
            if key not in ['model_name', 'task', 'view'] and not isinstance(value, dict):
                html += f"""
                    <tr>
                        <th>{key}</th>
                        <td>{value}</td>
                    </tr>
                """
        
        html += """
                </table>
            </div>
            
            <div class="metrics-summary">
                <h3>Performance Metrics</h3>
                <table>
        """
        
        # Add test metrics
        for key, value in tracker.test_metrics.items():
            if not isinstance(value, dict):
                html += f"""
                    <tr>
                        <th>{key}</th>
                        <td>{value:.4f if isinstance(value, float) else value}</td>
                    </tr>
                """
        
        html += """
                </table>
            </div>
            
            <div class="plots">
        """
        
        # Add training progress plot if available
        if train_progress_fig:
            train_progress_img = fig_to_base64(train_progress_fig)
            html += f"""
                <div class="plot">
                    <h3>Training Progress</h3>
                    <img src="data:image/png;base64,{train_progress_img}" alt="Training Progress">
                </div>
            """
            plt.close(train_progress_fig)
        
        # Add ROC curve if available
        if roc_fig:
            roc_img = fig_to_base64(roc_fig)
            html += f"""
                <div class="plot">
                    <h3>ROC Curve</h3>
                    <img src="data:image/png;base64,{roc_img}" alt="ROC Curve">
                </div>
            """
            plt.close(roc_fig)
        
        # Add other plots if available
        for plot_file in ['confusion_matrix.png', 'precision_recall_curve.png']:
            plot_path = os.path.join(model_dir, plot_file)
            if os.path.exists(plot_path):
                plot_name = ' '.join(plot_file.replace('.png', '').split('_')).title()
                with open(plot_path, 'rb') as f:
                    plot_img = base64.b64encode(f.read()).decode('utf-8')
                
                html += f"""
                    <div class="plot">
                        <h3>{plot_name}</h3>
                        <img src="data:image/png;base64,{plot_img}" alt="{plot_name}">
                    </div>
                """
        
        html += """
            </div>
        </div>
        <hr>
        """
        
        return html
    
    except Exception as e:
        return f"""
        <div class="model-section error">
            <h2>Error Loading Model: {os.path.basename(model_dir)}</h2>
            <p>Error: {str(e)}</p>
        </div>
        <hr>
        """


def generate_comparison_section(model_dirs):
    """
    Generate HTML content for model comparison.
    
    Args:
        model_dirs (list): List of model directory paths
        
    Returns:
        str: HTML content for the comparison section
    """
    html = """
    <div class="comparison-section">
        <h2>Model Comparison</h2>
    """
    
    try:
        # Create comparison table
        df = create_model_comparison_table(results_dir=None, model_dirs=model_dirs)
        
        # Clean up the table for display
        display_cols = ['model', 'task', 'view', 'test_auc', 'test_loss']
        for col in df.columns:
            if col.startswith('test_'):
                display_cols.append(col)
        
        display_df = df[display_cols] if all(col in df.columns for col in display_cols) else df
        
        # Convert DataFrame to HTML table
        table_html = display_df.to_html(classes='comparison-table', index=False)
        html += f"""
        <div class="comparison-table-container">
            {table_html}
        </div>
        """
        
        # Create comparison plots
        # Performance dashboard
        try:
            dashboard_fig = create_model_performance_dashboard(
                results_dir=None,
                model_dirs=model_dirs
            )
            
            dashboard_img = fig_to_base64(dashboard_fig)
            html += f"""
            <div class="plot">
                <h3>Performance Dashboard</h3>
                <img src="data:image/png;base64,{dashboard_img}" alt="Performance Dashboard">
            </div>
            """
            plt.close(dashboard_fig)
        except Exception as e:
            html += f"""
            <div class="error">
                <p>Error creating performance dashboard: {str(e)}</p>
            </div>
            """
        
        # ROC curve comparison
        try:
            roc_fig = plt.figure(figsize=(10, 6))
            plt.plot([0, 1], [0, 1], 'k--')
            
            for model_dir in model_dirs:
                y_true_path = os.path.join(model_dir, 'test_true.npy')
                y_score_path = os.path.join(model_dir, 'test_scores.npy')
                
                if os.path.exists(y_true_path) and os.path.exists(y_score_path):
                    y_true = np.load(y_true_path)
                    y_score = np.load(y_score_path)
                    
                    from sklearn.metrics import roc_curve
                    fpr, tpr, _ = roc_curve(y_true, y_score)
                    roc_auc = np.trapz(tpr, fpr)
                    
                    model_name = os.path.basename(model_dir).split('_')[0]
                    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
            
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves Comparison')
            plt.legend(loc='lower right')
            plt.grid(True)
            
            roc_img = fig_to_base64(roc_fig)
            html += f"""
            <div class="plot">
                <h3>ROC Curves Comparison</h3>
                <img src="data:image/png;base64,{roc_img}" alt="ROC Curves Comparison">
            </div>
            """
            plt.close(roc_fig)
        except Exception as e:
            html += f"""
            <div class="error">
                <p>Error creating ROC curve comparison: {str(e)}</p>
            </div>
            """
    
    except Exception as e:
        html += f"""
        <div class="error">
            <p>Error generating comparison section: {str(e)}</p>
        </div>
        """
    
    html += """
    </div>
    """
    
    return html


def generate_dashboard(model_dirs, output_file):
    """
    Generate the HTML dashboard.
    
    Args:
        model_dirs (list): List of model directory paths
        output_file (str): Path to save the HTML dashboard
    """
    # Generate HTML content
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MRNet Model Performance Dashboard</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            h1 {
                text-align: center;
                color: #3366cc;
            }
            h2 {
                color: #3366cc;
                border-bottom: 1px solid #cccccc;
                padding-bottom: 5px;
            }
            h3 {
                color: #3366cc;
            }
            .model-section, .comparison-section {
                margin-bottom: 30px;
                padding: 15px;
                background-color: #f9f9f9;
                border-radius: 5px;
            }
            .model-info, .metrics-summary {
                display: inline-block;
                vertical-align: top;
                margin-right: 30px;
            }
            table {
                border-collapse: collapse;
                margin: 15px 0;
            }
            th, td {
                padding: 8px 12px;
                border: 1px solid #ddd;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
            .plots {
                margin-top: 20px;
            }
            .plot {
                display: inline-block;
                margin: 10px;
                vertical-align: top;
            }
            .plot img {
                max-width: 550px;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
            .error {
                color: #cc0000;
                background-color: #ffeeee;
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
            }
            .comparison-table-container {
                overflow-x: auto;
            }
            .comparison-table {
                width: 100%;
            }
            hr {
                margin: 30px 0;
                border: 0;
                height: 1px;
                background-color: #ddd;
            }
        </style>
    </head>
    <body>
        <h1>MRNet Model Performance Dashboard</h1>
        
        <div class="dashboard-container">
    """
    
    # Add comparison section if multiple models
    if len(model_dirs) > 1:
        html += generate_comparison_section(model_dirs)
        html += "<hr>"
    
    # Add individual model sections
    html += "<h2>Individual Model Details</h2>"
    for model_dir in model_dirs:
        html += generate_model_section(model_dir)
    
    # Close HTML
    html += """
        </div>
        
        <div class="footer">
            <p>Generated on """ + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
        </div>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(output_file, 'w') as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(description='Generate HTML dashboard of MRNet model performance')
    parser.add_argument('--results_dir', type=str, required=True,
                      help='Directory containing model results')
    parser.add_argument('--output_file', type=str, default='dashboard.html',
                      help='Output HTML file path')
    parser.add_argument('--task', type=str, choices=['abnormal', 'acl', 'meniscus'],
                      help='Filter models by task')
    parser.add_argument('--view', type=str, choices=['axial', 'coronal', 'sagittal'],
                      help='Filter models by view')
    parser.add_argument('--backbone', type=str, choices=['alexnet', 'resnet18', 'densenet121'],
                      help='Filter models by backbone architecture')
    args = parser.parse_args()
    
    # Ensure results directory exists
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory '{args.results_dir}' does not exist")
        return 1
    
    # Find all model result directories
    from src.utils.compare_models import find_model_results, filter_models_by_criteria
    
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
    
    print(f"Selected {len(filtered_dirs)} models for dashboard:")
    for model_dir in filtered_dirs:
        print(f"  - {os.path.basename(model_dir)}")
    
    # Generate dashboard
    generate_dashboard(filtered_dirs, args.output_file)
    
    print(f"Dashboard saved to '{args.output_file}'")
    return 0


if __name__ == "__main__":
    exit(main()) 