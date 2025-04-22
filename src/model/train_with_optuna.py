import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score
import gc
import time
from torch.amp import autocast, GradScaler
import psutil
import optuna
from optuna.integration import TorchDistributedTrial
from optuna.samplers import TPESampler
import json

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Import the components we have created
from src.data_loader import MRNetDataset, SimpleMRIAugmentation
from src.model.MRNetModel import MRNetModel
from src.utils.metric_tracker import MetricTracker

def get_project_root():
    """Returns the absolute path to the project root directory"""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
            print(f"Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    else:
        print("WARNING: CUDA is not available. Multi-GPU training cannot proceed.")
    print("=====================\n")

def print_gpu_memory_stats(rank, location):
    """
    Print GPU memory usage at a specific location in the code
    
    Args:
        rank (int): The process rank (GPU ID)
        location (str): Description of where in the code this function is called
    """
    if rank == 0:  # Only print from master process
        torch.cuda.synchronize()
        gc.collect()
        memory_allocated = torch.cuda.memory_allocated(rank) / (1024 ** 3)
        memory_reserved = torch.cuda.memory_reserved(rank) / (1024 ** 3)
        max_memory_allocated = torch.cuda.max_memory_allocated(rank) / (1024 ** 3)
        
        print(f"\n=== GPU Memory Stats at {location} ===")
        print(f"GPU {rank} - Currently allocated: {memory_allocated:.2f} GB")
        print(f"GPU {rank} - Currently reserved: {memory_reserved:.2f} GB")
        print(f"GPU {rank} - Max allocated: {max_memory_allocated:.2f} GB")
        print("===================================\n")

def setup_ddp(rank, world_size, args):
    """
    Setup for Distributed Data Parallel
    
    Args:
        rank (int): Unique ID for each process
        world_size (int): Total number of processes
        args (argparse.Namespace): Command line arguments
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.master_port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    if rank == 0:
        print(f"Process {rank}: DDP setup complete, using device cuda:{rank}")

def cleanup_ddp():
    """
    Cleanup distributed processes
    """
    dist.destroy_process_group()

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
            # Skip batches where the required view is not available
            if args.view not in batch['available_views']:
                continue
                
            data = batch[args.view].to(device)
            
            # Get labels
            labels = batch['label'].to(device)
            labels = labels.view(-1, 1)
            
            # Forward pass with automatic mixed precision
            with autocast(device_type='cuda', enabled=True):
                outputs = model(data)
            
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

def run_optimization(rank, world_size, args):
    """
    Run Optuna optimization for hyperparameter tuning
    
    Args:
        rank (int): Process rank
        world_size (int): Total number of processes
        args (argparse.Namespace): Command line arguments
    """
    # Setup DDP first
    setup_ddp(rank, world_size, args)
    
    is_master = rank == 0
    
    # Only create and manage the study in the master process
    study = None
    if is_master:
        # Create Optuna storage
        storage_name = os.path.join(args.output_dir, "optuna_study.db")
        storage_url = f"sqlite:///{storage_name}"
        
        # Create a sampler with multivariate TPE
        sampler = TPESampler(multivariate=True, seed=42)
        
        # Create or load the study
        try:
            study = optuna.create_study(
                study_name=f"mrnet_{args.task}_{args.view}",
                storage=storage_url,
                load_if_exists=True,
                sampler=sampler,
                direction="maximize"  # We want to maximize AUC
            )
            print(f"Study loaded: {study.study_name}")
        except Exception as e:
            print(f"Error creating/loading study: {e}")
            # Try without storage as fallback
            study = optuna.create_study(
                direction="maximize",
                sampler=sampler
            )
            print("Created study without persistent storage")
    
    # Synchronize all processes
    dist.barrier()
    
    # Run optimization only in the master process
    if is_master:
        print("\n=== Starting Hyperparameter Optimization ===")
        print(f"Running {args.n_trials} trials for {args.task} task, {args.view} view")
        
        try:
            # Create a wrapper function that doesn't reinitialize DDP
            def objective_wrapper(trial):
                return objective(trial, rank, world_size, args, is_ddp_initialized=True)
            
            study.optimize(
                objective_wrapper,
                n_trials=args.n_trials,
                timeout=args.timeout
            )
            
            # Print optimization summary
            print("\n=== Optimization Results ===")
            print(f"Best trial: {study.best_trial.number}")
            print(f"Best value (AUC): {study.best_trial.value:.4f}")
            print("\nBest hyperparameters:")
            for key, value in study.best_trial.params.items():
                print(f"  {key}: {value}")
                
            # Save best parameters to a JSON file
            best_params_path = os.path.join(args.output_dir, "best_params.json")
            with open(best_params_path, 'w') as f:
                json.dump(study.best_trial.params, f, indent=2)
            print(f"\nBest parameters saved to {best_params_path}")
            
            # Plot optimization results
            try:
                from optuna.visualization import plot_optimization_history, plot_param_importances
                
                # Plot optimization history
                fig = plot_optimization_history(study)
                fig.write_image(os.path.join(args.output_dir, "optimization_history.png"))
                
                # Plot parameter importances
                fig = plot_param_importances(study)
                fig.write_image(os.path.join(args.output_dir, "param_importances.png"))
                print(f"Visualization plots saved to {args.output_dir}")
            except Exception as e:
                print(f"Error creating visualization plots: {e}")
            
        except Exception as e:
            print(f"Error during optimization: {e}")
            import traceback
            traceback.print_exc()
    else:
        # Non-master processes run the same number of trials but results are discarded
        for _ in range(args.n_trials):
            try:
                # We create a dummy trial
                dummy_trial = optuna.trial.Trial(study, 0)
                objective(dummy_trial, rank, world_size, args, is_ddp_initialized=True)
            except optuna.TrialPruned:
                continue
            except Exception as e:
                print(f"Error in worker process {rank}: {e}")
                break
    
    # Synchronize all processes before exiting
    dist.barrier()
    
    # Cleanup DDP
    cleanup_ddp()

def objective(trial, rank, world_size, args, is_ddp_initialized=False):
    """
    Optuna objective function for hyperparameter optimization
    
    Args:
        trial: Optuna trial object
        rank (int): Process rank
        world_size (int): Total number of processes
        args (argparse.Namespace): Command line arguments
        is_ddp_initialized (bool): Whether DDP is already initialized
        
    Returns:
        float: Validation AUC score (higher is better)
    """
    # Only setup DDP if not already initialized
    if not is_ddp_initialized:
        setup_ddp(rank, world_size, args)
    
    device = torch.device(f'cuda:{rank}')
    
    # Only print from master process
    is_master = rank == 0
    
    # For distributed training, we use TorchDistributedTrial
    if world_size > 1:
        trial = TorchDistributedTrial(trial, rank == 0)
    
    # Sample hyperparameters for this trial
    if is_master:
        print(f"\n=== Trial {trial.number} ===")
    
    # Sample hyperparameters from the search space
    backbone = trial.suggest_categorical('backbone', ['resnet18', 'resnet34', 'densenet121'])
    batch_size = trial.suggest_categorical('batch_size', [2, 4, 6])
    lr = trial.suggest_float('lr', 1e-6, 1e-4, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    max_slices = trial.suggest_categorical('max_slices', [24,32, 48])
    #dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    use_augmentation = trial.suggest_categorical('use_augmentation', [True, False])
    
    # Print hyperparameters for this trial
    if is_master:
        print(f"Hyperparameters for Trial {trial.number}:")
        print(f"  Backbone: {backbone}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Learning Rate: {lr:.6f}")
        print(f"  Weight Decay: {weight_decay:.6f}")
        print(f"  Max Slices: {max_slices}")
        #print(f"  Dropout Rate: {dropout_rate:.2f}")
        print(f"  Use Augmentation: {use_augmentation}")
    
    # Initialize TensorBoard writer (only on master process)
    writer = None
    if is_master:
        log_dir = os.path.join(args.output_dir, f'logs/trial_{trial.number}')
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
    
    # Initialize MetricTracker
    metric_tracker = None
    if is_master:
        model_name = f"{backbone}_{args.task}_{args.view}_trial{trial.number}"
        config = {
            'backbone': backbone,
            'batch_size': batch_size,
            'lr': lr,
            'weight_decay': weight_decay,
            'max_slices': max_slices,
            'use_augmentation': use_augmentation,
            'task': args.task,
            'view': args.view,
            'trial': trial.number
        }
        metric_tracker = MetricTracker(
            model_name=model_name,
            task=args.task,
            view=args.view,
            config=config,
            output_dir=os.path.join(args.output_dir, f'trial_{trial.number}')
        )
    
    # Create datasets with updated parameters
    augmentation = SimpleMRIAugmentation(p=0.5) if use_augmentation else None
    
    # Create train dataset
    train_dataset = MRNetDataset(
        root_dir=project_root,
        task=args.task,
        split='train',
        transform=augmentation,
        max_slices=max_slices,
        view=args.view
    )
    
    # Create validation dataset
    valid_dataset = MRNetDataset(
        root_dir=project_root,
        task=args.task,
        split='valid',
        transform=None,
        max_slices=max_slices,
        view=args.view
    )
    
    # Create distributed sampler for training data
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate,
        pin_memory=True,
        sampler=train_sampler
    )
    
    # Validation loader doesn't need to be distributed
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate,
        pin_memory=True
    )
    
    # Create model with the selected hyperparameters
    model = MRNetModel(backbone=backbone)
    
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
    model = model.to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=is_master
    )
    
    # Loss function
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Enable mixed precision training
    use_amp = True
    scaler = GradScaler('cuda') if use_amp else None
    
    # Training loop variables
    best_val_auc = 0.0
    trial_dir = os.path.join(args.output_dir, f'trial_{trial.number}')
    os.makedirs(trial_dir, exist_ok=True)
    best_model_path = os.path.join(trial_dir, f"best_model_{backbone}_{args.task}_{args.view}.pth")
    global_step = 0
    
    # Train for a smaller number of epochs in each trial
    num_epochs = min(args.epochs, 10)  # Limit to 10 epochs per trial to save time
    
    # Training loop
    for epoch in range(num_epochs):
        if is_master:
            print(f"\n[Trial {trial.number}] Starting epoch {epoch+1}/{num_epochs}")
        
        # Set epoch for sampler
        train_sampler.set_epoch(epoch)
        
        # Training phase
        model.train()
        running_loss = 0.0
        train_true = []
        train_pred = []
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                # Ensure the required view is available
                if args.view not in batch['available_views']:
                    continue
                
                # Get data for this view
                data = batch[args.view].to(device)
                
                # Get labels and reshape to [batch_size, 1]
                labels = batch['label'].to(device)
                labels = labels.view(-1, 1)
                
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
                
                # Log metrics from master process
                log_interval = 10
                if is_master and batch_idx % log_interval == 0:
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
                
                global_step += 1
                
                # Clean up memory
                torch.cuda.empty_cache()
                
            except Exception as e:
                if is_master:
                    print(f"\n=== ERROR in batch {batch_idx+1} ===")
                    print(f"Error type: {type(e).__name__}")
                    print(f"Error message: {str(e)}")
                
                # Report error to Optuna and skip this trial
                if not is_ddp_initialized:
                    cleanup_ddp()
                return 0.0  # Return 0 AUC to mark this trial as failed
        
        # End of epoch - calculate final training metrics
        train_loss = running_loss / len(train_loader)
        train_auc = roc_auc_score(train_true, train_pred) if len(np.unique(train_true)) > 1 else 0.0
        
        # Evaluate on validation set
        val_loss, val_auc, val_true, val_pred = evaluate(model, device, valid_loader, criterion, args)
        
        # Report intermediate value to Optuna
        trial.report(val_auc, epoch)
        
        # Log validation metrics
        if is_master:
            print(f"\n[Trial {trial.number}, Epoch {epoch+1}/{num_epochs}] "
                  f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")
            
            # Log to TensorBoard
            if writer:
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('AUC/val', val_auc, epoch)
            
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
                torch.save(model.module.state_dict(), best_model_path)
                print(f"[Trial {trial.number}] New best model saved with Val AUC: {val_auc:.4f}")
        
        # Adjust learning rate using the scheduler
        scheduler.step(val_loss)
        
        # Check if the trial should be pruned
        if trial.should_prune():
            if is_master:
                print(f"[Trial {trial.number}] Pruned at epoch {epoch+1}")
            if not is_ddp_initialized:
                cleanup_ddp()
            raise optuna.TrialPruned()
    
    # End of training - cleanup
    if is_master:
        print(f"\n[Trial {trial.number}] Training completed | Best Val AUC: {best_val_auc:.4f}")
        
        # Generate training plots if we have a metric tracker
        if metric_tracker:
            metric_tracker.save_metrics()
        
        # Close TensorBoard writer
        if writer:
            writer.close()
    
    # Only cleanup DDP if we initialized it in this function
    if not is_ddp_initialized:
        cleanup_ddp()
    
    # Return the best validation AUC (Optuna maximizes this value)
    return best_val_auc

def main():
    parser = argparse.ArgumentParser(description='MRNet Hyperparameter Optimization with Optuna')
    
    # Data parameters
    parser.add_argument('--task', type=str, default='abnormal',
                        choices=['abnormal', 'acl', 'meniscus'],
                        help='Task to train on')
    parser.add_argument('--view', type=str, required=True,
                        choices=['axial', 'coronal', 'sagittal'],
                        help='MRI view to train on')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10,
                        help='Maximum number of epochs to train each trial')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of workers for data loading')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='optuna_results',
                        help='Directory to save optimization results')
    
    # Multi-GPU specific parameters
    parser.add_argument('--master_port', type=str, default='12355',
                        help='Port for distributed training')
    parser.add_argument('--sync_bn', action='store_true',
                        help='Use synchronized batch normalization')
    parser.add_argument('--num_gpus', type=int, default=None,
                        help='Number of GPUs to use (default: all available)')
    
    # Optuna parameters
    parser.add_argument('--n_trials', type=int, default=20,
                        help='Number of optimization trials to run')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Time limit for optimization in seconds')
    
    args = parser.parse_args()
    
    # Set train_approach to per_view for simplicity
    args.train_approach = 'per_view'
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get number of available GPUs
    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        raise ValueError("No GPUs are available. Multi-GPU training cannot proceed.")
    
    # Decide how many GPUs to use
    num_gpus = args.num_gpus if args.num_gpus is not None else available_gpus
    num_gpus = min(num_gpus, available_gpus)
    
    print(f"Starting hyperparameter optimization on {num_gpus} GPUs")
    
    # Launch distributed optimization
    mp.spawn(
        run_optimization,
        args=(num_gpus, args),
        nprocs=num_gpus,
        join=True
    )

if __name__ == "__main__":
    main() 