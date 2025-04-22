import os
import sys
import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score
import gc
import time
from torch.amp import autocast, GradScaler
import psutil

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# This is importing the components we have created
from src.data_loader import MRNetDataset, SimpleMRIAugmentation
from src.model.MRNetModel import MRNetModel
from src.utils.metric_tracker import MetricTracker

# Add EarlyStopping class
class EarlyStopping:
    """
    Early stopping to stop training when validation performance doesn't improve for a specified
    number of consecutive epochs (patience).
    
    Args:
        patience (int): How many epochs to wait after last improvement
        verbose (bool): If True, prints a message for each validation improvement
        delta (float): Minimum change to qualify as an improvement
        mode (str): 'min' for loss, 'max' for metrics like AUC
    """
    def __init__(self, patience=7, verbose=True, delta=0, mode='max'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.mode = mode
        if mode == 'min':
            self.val_score_min = float('inf')
        else:  # mode == 'max'
            self.val_score_min = float('-inf')
        
    def __call__(self, val_score, model=None, path=None):
        score = val_score
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, model, path)
        else:
            if self.mode == 'min':
                improve_condition = score < self.best_score - self.delta
            else:  # mode == 'max'
                improve_condition = score > self.best_score + self.delta
                
            if improve_condition:
                self.best_score = score
                self.save_checkpoint(val_score, model, path)
                self.counter = 0
            else:
                self.counter += 1
                if self.verbose:
                    print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
        
        return self.early_stop
    
    def save_checkpoint(self, val_score, model=None, path=None):
        '''Save model when validation metric improves.'''
        if self.verbose:
            metric_name = "loss" if self.mode == 'min' else "metric"
            print(f'Validation {metric_name} improved ({self.val_score_min:.6f} --> {val_score:.6f})')
            
        if self.mode == 'min':
            self.val_score_min = val_score
        else:  # mode == 'max'
            self.val_score_min = val_score
            
        if model is not None and path is not None:
            # Save model state dict (modified to handle non-DDP models)
            if hasattr(model, 'module'):
                torch.save(model.module.state_dict(), path)
            else:
                torch.save(model.state_dict(), path)

def get_project_root():
    """Returns the absolute path to the project root directory"""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


#checks gpu availability
def verify_gpu():
    """
    Verify GPU availability and print information about available GPUs
    """
    print("\n=== GPU Information ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("WARNING: CUDA is not available. GPU training cannot proceed.")
    print("=====================\n")


#main training function
def train_model(args):
    """
    Training function for single GPU training
    
    Args:
        args (argparse.Namespace): Command line arguments
    
    Returns:
        float: Best validation AUC achieved during training
    """
    try:
        # Set up device
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available. GPU training cannot proceed.")
        
        device = torch.device('cuda:0')
        torch.cuda.set_device(0)
        
        # Enable mixed precision training to reduce memory usage
        use_amp = True
        scaler = GradScaler() if use_amp else None
        
        # Initialize TensorBoard writer
        log_dir = os.path.join(args.output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logs will be saved to {log_dir}")
        
        # Initialize MetricTracker
        model_name = f"{args.backbone}_{args.task}_{args.view}"
        config = vars(args)  # Convert args to dictionary
        metric_tracker = MetricTracker(
            model_name=model_name,
            task=args.task,
            view=args.view,
            config=config,
            output_dir=args.output_dir
        )
        print(f"Metrics will be tracked and saved to {args.output_dir}")
        
        # Initialize early stopping
        early_stopping = None
        if args.early_stopping:
            early_stopping = EarlyStopping(
                patience=args.early_stopping_patience, 
                verbose=True, 
                delta=args.early_stopping_delta,
                mode='max'  # Using 'max' for AUC metric
            )
            print(f"Early stopping enabled with patience={args.early_stopping_patience}, delta={args.early_stopping_delta}")

        # Print initial training process
        print(f"\n=== Training Process Initialization ===")
        print(f"Starting training on GPU")
        print(f"Using mixed precision training: {use_amp}")
        print(f"Running with PyTorch version: {torch.__version__}")
        verify_gpu()
        
        # Get project root and create output directory
        project_root = get_project_root()
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Print configuration
        print(f"\n=== Configuration ===")
        print(f"Project root: {project_root}")
        print(f"Output directory: {args.output_dir}")
        print(f"Task: {args.task}")
        print(f"View: {args.view}")
        print(f"Training approach: {args.train_approach}")
        print(f"Backbone: {args.backbone}")
        print(f"Batch size: {args.batch_size}")
        print(f"Learning rate: {args.lr}")
        print(f"Weight decay: {args.weight_decay}")
        print(f"Epochs: {args.epochs}")
        print(f"Workers: {args.num_workers}")
        
        # Create datasets
        print(f"Creating train dataset for task: {args.task}")
        
        # Determine which view to load
        view_to_load = args.view
        
        # Determine max slices based on backbone model
        if args.max_slices is not None:
            max_slices = args.max_slices  # Use user-specified value
        elif args.backbone == 'densenet121':
            max_slices = 32  # Use fewer slices for DenseNet121
        else:  # resnet18 or others
            max_slices = 64  # Use more slices for other backbones
            
        print(f"Using {max_slices} max slices for {args.backbone}")
        
        # Create datasets with updated parameters
        start_time = time.time()
        root = args.data_dir or get_project_root()
        train_dataset = MRNetDataset(
            root_dir=root,
            task=args.task,
            split='train',
            transform=SimpleMRIAugmentation(p=0.5) if args.use_augmentation else None,
            max_slices=max_slices,
            view=view_to_load  # Pass the view to load
        )
        print(f"Train dataset created in {time.time() - start_time:.2f} seconds")
        print(f"Creating validation dataset")
        
        start_time = time.time()
        valid_dataset = MRNetDataset(
            root_dir=root,
            task=args.task,
            split='valid',
            transform=None,
            max_slices=max_slices,
            view=view_to_load  # Pass the view to load
        )
        print(f"Validation dataset created in {time.time() - start_time:.2f} seconds")
        
        # Create data loaders
        print(f"Creating data loaders with batch size {args.batch_size} and {args.num_workers} workers")
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,  # Now we can shuffle without DistributedSampler
            num_workers=args.num_workers,
            collate_fn=custom_collate,
            pin_memory=True
        )
        
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=custom_collate,
            pin_memory=True
        )
        
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(valid_dataset)}")
        print(f"Train loader batches: {len(train_loader)}")
        print(f"Validation loader batches: {len(valid_loader)}")
        
        # Create model for the specified view
        models = {}
        optimizers = {}
        schedulers = {}
        
        # Create model
        print(f"Creating model for view: {args.view}")
        start_time = time.time()
        model = MRNetModel(backbone=args.backbone)
        model = model.to(device)
        
        # Create optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Store the model, optimizer, and scheduler
        models[args.view] = model
        optimizers[args.view] = optimizer
        schedulers[args.view] = scheduler
        
        print(f"Model for {args.view} created in {time.time() - start_time:.2f} seconds")
        
        # Loss function
        criterion = torch.nn.BCEWithLogitsLoss()
        
        # Training loop
        best_val_auc = 0.0
        best_model_path = os.path.join(args.output_dir, f"best_model_{args.backbone}_{args.task}_{args.view}.pth")
        global_step = 0
        
        print("\n=== Training Configuration ===")
        print(f"Task: {args.task}")
        print(f"View: {args.view}")
        print(f"Training Approach: {args.train_approach}")
        print(f"Backbone: {args.backbone}")
        print(f"Batch Size: {args.batch_size}")
        print(f"Learning Rate: {args.lr}")
        print(f"Weight Decay: {args.weight_decay}")
        print(f"Number of Epochs: {args.epochs}")
        print(f"Number of Workers: {args.num_workers}")
        print(f"Data Augmentation: {'Enabled' if args.use_augmentation else 'Disabled'}")
        print(f"Max Slices per MRI: {max_slices}")
        print(f"Mixed Precision Training: {'Enabled' if use_amp else 'Disabled'}")
        print(f"Early Stopping: {'Enabled' if args.early_stopping else 'Disabled'}")
        if args.early_stopping:
            print(f"  - Patience: {args.early_stopping_patience} epochs")
            print(f"  - Delta: {args.early_stopping_delta}")
        print("==============================\n")
        
        # Main training loop
        for epoch in range(args.epochs):
            print(f"\nStarting epoch {epoch+1}/{args.epochs}")
            
            
            # Training phase
            model = models[args.view]
            model.train()
            optimizer = optimizers[args.view]
            running_loss = 0.0
            train_true = []
            train_pred = []
            
            for batch_idx, batch in enumerate(train_loader):
                try:
                    # Ensure the required view is available
                    if args.view not in batch['available_views']:
                        if batch_idx % 20 == 0:
                            print(f"Skipping batch {batch_idx+1} - view {args.view} not available")
                        continue
                    
                    batch_start_time = time.time()
                    batch_size = batch['label'].size(0)
                    
                    # Print detailed batch info for early batches
                    if batch_idx < 5:
                        print(f"\n=== Batch {batch_idx+1} Details ===")
                        print(f"Batch keys: {list(batch.keys())}")
                        print(f"Available views: {batch['available_views']}")
                        print(f"Label shape: {batch['label'].shape}")
                        for view in batch['available_views']:
                            print(f"{view} shape: {batch[view].shape}")
                            # Calculate approximate memory usage
                            tensor_size_mb = batch[view].element_size() * batch[view].nelement() / (1024 * 1024)
                            print(f"{view} memory: {tensor_size_mb:.2f} MB")
                    
                    # Get labels and reshape to [batch_size, 1]
                    labels = batch['label'].to(device)
                    labels = labels.view(-1, 1)  # Reshape labels to match output
                    
                    # Get the data for this view
                    data = batch[args.view].to(device)
                    
                    # Log shape information
                    if batch_idx % 10 == 0:
                        print(f"Processing {args.view} data with shape: {data.shape}")
                    
                    # Zero gradients
                    optimizer.zero_grad()
                    
                    # Forward pass with automatic mixed precision
                    with autocast(device_type='cuda', enabled=use_amp):
                        outputs = model(data)
                    
                    # Calculate loss
                    loss = criterion(outputs, labels)
                    
                    # Backward pass with gradient scaling
                    if use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                    
                    
                    running_loss += loss.item()
                    
                    # Store predictions and labels for AUC calculation
                    with torch.no_grad():
                        train_pred.extend(torch.sigmoid(outputs).cpu().numpy())
                        train_true.extend(labels.cpu().numpy())
                    
                    # Log metrics
                    log_interval = getattr(args, 'log_interval', 10)  # Default to 10 if not set
                    if batch_idx % log_interval == 0:
                        avg_loss = running_loss / (batch_idx + 1)
                        
                        # Calculate training AUC if we have enough samples
                        train_auc = 0.0
                        if len(train_true) > 1 and len(np.unique(train_true)) > 1:
                            train_auc = roc_auc_score(train_true, train_pred)
                        
                        # Log to TensorBoard
                        if writer:
                            writer.add_scalar('Loss/train', avg_loss, global_step)
                            if train_auc > 0:
                                writer.add_scalar('AUC/train', train_auc, global_step)
                        
                        # Update MetricTracker
                        if metric_tracker:
                            metrics = {'loss': avg_loss}
                            if train_auc > 0:
                                metrics['auc'] = train_auc
                            metric_tracker.update_train(metrics, global_step)
                        
                        # Print progress
                        print(f"Epoch {epoch}/{args.epochs} | Batch {batch_idx}/{len(train_loader)} | "
                              f"Loss: {avg_loss:.4f}" + (f" | AUC: {train_auc:.4f}" if train_auc > 0 else ""))
                    
                    global_step += 1
                    
                    # Call gc.collect() and empty cache occasionally to keep memory usage in check
                    if batch_idx % 50 == 0:
                        torch.cuda.empty_cache()
                        gc.collect()
                    
                except Exception as e:
                    print(f"\n=== ERROR in batch {batch_idx+1} ===")
                    print(f"Error type: {type(e).__name__}")
                    print(f"Error message: {str(e)}")
                    print(f"Model: {args.view}")
                    print(f"Batch keys: {list(batch.keys()) if 'batch' in locals() else 'Unknown'}")
                    print(f"Available views: {batch['available_views'] if 'batch' in locals() and 'available_views' in batch else 'Unknown'}")
                    
                    # Print stack trace
                    import traceback
                    print("\nStack trace:")
                    traceback.print_exc()
                    
                    # Try to suggest potential fixes
                    print("\nPossible fixes to try:")
                    if "CUDA out of memory" in str(e):
                        print("1. Reduce batch size further")
                        print("2. Reduce max_slices value in MRNetDataset")
                        print("3. Use a smaller backbone model")
                        print("4. Implement gradient accumulation (effective batch size without memory increase)")
                    elif "dimension out of range" in str(e) or "size mismatch" in str(e):
                        print("1. Check tensor shapes throughout the pipeline")
                        print("2. Ensure all MRIs in a batch have compatible dimensions")
                        print("3. Review model architecture for size compatibility")
                    
                    # Re-raise to stop training or handle as needed
                    raise e
            
            # End of epoch - calculate final training metrics
            train_loss = running_loss / len(train_loader)
            train_auc = roc_auc_score(train_true, train_pred) if len(np.unique(train_true)) > 1 else 0.0
            
            # Evaluate on validation set
            val_loss, val_auc, val_true, val_pred = evaluate(model, device, valid_loader, criterion, args)
            
            # Log validation metrics
            print(f"\nEpoch {epoch}/{args.epochs} completed | "
                  f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")
            
            # Log to TensorBoard
            if writer:
                writer.add_scalar('Loss/val', val_loss, global_step)
                writer.add_scalar('AUC/val', val_auc, global_step)
            
            # Update MetricTracker with validation metrics
            if metric_tracker:
                val_metrics = {
                    'loss': val_loss,
                    'auc': val_auc
                }
                metric_tracker.update_val(val_metrics, epoch)
            
            # Save the best model
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved with Val AUC: {val_auc:.4f}")
            
            # Save epoch predictions for later analysis (only for best epoch)
            if val_auc == best_val_auc and metric_tracker:
                np.save(os.path.join(args.output_dir, 'val_true.npy'), np.array(val_true))
                np.save(os.path.join(args.output_dir, 'val_pred.npy'), np.array(val_pred))
            
            # Adjust learning rate using the scheduler
            scheduler = schedulers[args.view]
            scheduler.step(val_loss)
            
            # Check for early stopping
            if early_stopping and early_stopping(val_auc, model, best_model_path):
                print(f"Early stopping triggered after {epoch+1} epochs. Best Val AUC: {best_val_auc:.4f}")
                break
        
        # End of training - cleanup and save final metrics
        print(f"\nTraining completed | Best Val AUC: {best_val_auc:.4f}")
        
        # Generate training plots if we have a metric tracker
        if metric_tracker:
            metric_tracker.save_metrics()
            
            # Generate basic visualization of metrics
            from src.utils.visualization import create_training_progress_report
            try:
                # Create and save training progress report
                fig = create_training_progress_report(
                    args.output_dir, 
                    save_path=os.path.join(args.output_dir, 'training_progress.png')
                )
                print(f"Training progress report saved to {args.output_dir}")
            except Exception as e:
                print(f"Error generating training report: {str(e)}")
        
        # Close TensorBoard writer
        if writer:
            writer.close()
        
        return best_val_auc
    
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def evaluate(model, device, data_loader, criterion, args):
    """
    Evaluate model on validation data
    
    Args:
        model (nn.Module): PyTorch model
        device (torch.device): Device to run evaluation on
        data_loader (DataLoader): Validation data loader
        criterion (nn.Module): Loss function
        args (argparse.Namespace): Command line arguments
        
    Returns:
        tuple: (val_loss, val_auc, val_true, val_pred)
    """
    model.eval()
    val_loss = 0
    val_pred = []
    val_true = []
    
    with torch.no_grad():
        for batch in data_loader:
            # Get the appropriate data based on training approach
            if args.train_approach == 'per_view':
                if args.view not in batch['available_views']:
                    continue
                
                data = batch[args.view].to(device)
                with autocast(device_type='cuda', enabled=True):
                    outputs = model(data)
            else:  # ensemble approach
                data_dict = {view: batch[view].to(device) for view in batch['available_views']}
                with autocast(device_type='cuda', enabled=True):
                    outputs = model(data_dict)
            
            # Get labels
            labels = batch['label'].to(device)
            labels = labels.view(-1, 1)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # Store predictions and true labels
            val_pred.extend(torch.sigmoid(outputs).cpu().numpy())
            val_true.extend(labels.cpu().numpy())
    
    # Calculate metrics
    val_loss /= len(data_loader)
    val_auc = roc_auc_score(val_true, val_pred) if len(np.unique(val_true)) > 1 else 0.0
    
    return val_loss, val_auc, val_true, val_pred


#this might be a duplicate function, we need to evaluate if this is necessary
def custom_collate(batch):
    """
    Custom collate function for handling MRI data batches with variable sizes
    """
    if not batch:
        return {}
    
    # Extract all available keys from the first item
    keys = batch[0].keys()
    result = {}
    
    for key in keys:
        if key == 'label':
            # Stack all labels
            result[key] = torch.stack([item[key] for item in batch])
        elif key == 'available_views':
            # Gather all available views in this batch
            result[key] = []
            for item in batch:
                for view in item[key]:
                    if view not in result[key]:
                        result[key].append(view)
        elif key in ['axial', 'coronal', 'sagittal']:
            # For MRI views, only include if they exist in all samples
            valid_samples = [item[key] for item in batch if key in item.get('available_views', [])]
            if valid_samples:
                try:
                    result[key] = torch.stack(valid_samples)
                except:
                    # If tensors can't be stacked (different sizes), use a simple list
                    result[key] = valid_samples
        else:
            # For other keys, just collect as a list
            result[key] = [item[key] for item in batch if key in item]
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Train MRNet models on a single GPU')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Root directory of the project (default: project root)')
    parser.add_argument('--task', type=str, default='abnormal',
                        choices=['abnormal', 'acl', 'meniscus'],
                        help='Task to train on')
    parser.add_argument('--view', type=str, default=None,
                        choices=['axial', 'coronal', 'sagittal'],
                        help='MRI view to train on (for single-view training)')
    parser.add_argument('--max_slices', type=int, default=None,
                        help='Maximum number of slices to use per MRI volume (default: 32 for densenet121, 64 for others)')
    
    # Model parameters
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['alexnet', 'resnet18', 'resnet34', 'densenet121'],
                        help='Backbone architecture')
    parser.add_argument('--train_approach', type=str, default='per_view',
                        choices=['per_view'],  # Only per_view is available now
                        help='Training approach')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of workers for data loading')
    parser.add_argument('--use_augmentation', action='store_true',
                        help='Enable data augmentation during training')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='How often to log training metrics (in batches)')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='model_outputs',
                        help='Directory to save models and logs')
    
    # Add arguments for metrics tracking and visualization
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save predictions from validation and test sets for later analysis')
    parser.add_argument('--generate_plots', action='store_true',
                       help='Generate plots of training progress and model performance')
    
    # Add early stopping parameters
    parser.add_argument('--early_stopping', action='store_true',
                        help='Enable early stopping')
    parser.add_argument('--early_stopping_patience', type=int, default=2,
                        help='Number of epochs with no improvement after which training will be stopped')
    parser.add_argument('--early_stopping_delta', type=float, default=0.01,
                        help='Minimum change to qualify as an improvement')
    
    args = parser.parse_args()
    
    # Override train_approach to always be per_view
    args.train_approach = 'per_view'
    
    # Verify GPU availability
    if not torch.cuda.is_available():
        raise ValueError("No GPU is available. Training cannot proceed.")
    
    # Set data directory to project root if not specified
    if args.data_dir is None:
        args.data_dir = get_project_root()
    
    print(f"Starting training on GPU")
    
    # Launch training
    best_val_auc = train_model(args)
    print(f"Training completed with best validation AUC: {best_val_auc:.4f}")

if __name__ == "__main__":
    main()
