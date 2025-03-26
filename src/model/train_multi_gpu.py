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

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Import your existing components
from src.data_loader import MRNetDataset, SimpleMRIAugmentation
from src.model.MRNetModel import MRNetModel, MRNetEnsemble

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

def train_model_ddp(rank, world_size, args):
    """
    Training function for distributed training
    
    Args:
        rank (int): Process rank
        world_size (int): Total number of processes
        args (argparse.Namespace): Command line arguments
    """
    # Setup DDP
    try:
        setup_ddp(rank, world_size, args)
        device = torch.device(f'cuda:{rank}')
        
        # Enable mixed precision training to reduce memory usage
        use_amp = True
        scaler = GradScaler('cuda') if use_amp else None
        
        # Only print from master process
        is_master = rank == 0
        
        # Initialize TensorBoard writer (only on master process)
        writer = None
        if is_master:
            log_dir = os.path.join(args.output_dir, 'logs')
            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=log_dir)
            print(f"[Process {rank}] TensorBoard logs will be saved to {log_dir}")
        
        if is_master:
            print(f"\n=== Training Process Initialization ===")
            print(f"Process {rank}: Starting training on {world_size} GPUs")
            print(f"Using mixed precision training: {use_amp}")
            print(f"Running with PyTorch version: {torch.__version__}")
            verify_gpu()
            
            # Print CUDA memory at start
            print("\n=== Initial GPU Memory Stats ===")
            for i in range(torch.cuda.device_count()):
                free_mem, total_mem = torch.cuda.mem_get_info(i)
                free_mem_gb = free_mem / (1024**3)
                total_mem_gb = total_mem / (1024**3)
                print(f"GPU {i}: Free Memory: {free_mem_gb:.2f} GB / Total: {total_mem_gb:.2f} GB")
            
            print_gpu_memory_stats(rank, "after init")
        
        # Get project root and create output directory
        project_root = get_project_root()
        os.makedirs(args.output_dir, exist_ok=True)
        
        if is_master:
            print(f"\n=== Configuration ===")
            print(f"Project root: {project_root}")
            print(f"Output directory: {args.output_dir}")
            print(f"Task: {args.task}")
            print(f"View: {args.view}")
            print(f"Training approach: per_view")  # Hardcoded to per_view now
            print(f"Backbone: {args.backbone}")
            print(f"Batch size: {args.batch_size} per GPU")
            print(f"Total effective batch size: {args.batch_size * world_size}")
            print(f"Learning rate: {args.lr}")
            print(f"Weight decay: {args.weight_decay}")
            print(f"Epochs: {args.epochs}")
            print(f"Workers per GPU: {args.num_workers}")
        
        # Create datasets
        if is_master:
            print(f"[Process {rank}] Creating train dataset for task: {args.task}")
        
        start_time = time.time()
        # Determine which view to load if using per_view approach
        view_to_load = args.view
        
        # Determine max slices based on whether we're loading a single view
        max_slices = 64  # Use more slices if loading a single view
        
        # Create datasets with updated parameters
        train_dataset = MRNetDataset(
            root_dir=project_root,
            task=args.task,
            split='train',
            transform=SimpleMRIAugmentation(p=0.5) if args.use_augmentation else None,
            max_slices=max_slices,
            view=view_to_load  # Pass the view to load
        )
        if is_master:
            print(f"[Process {rank}] Train dataset created in {time.time() - start_time:.2f} seconds")
            print(f"[Process {rank}] Creating validation dataset")
        
        start_time = time.time()
        valid_dataset = MRNetDataset(
            root_dir=project_root,
            task=args.task,
            split='valid',
            transform=None,
            max_slices=max_slices,
            view=view_to_load  # Pass the view to load
        )
        if is_master:
            print(f"[Process {rank}] Validation dataset created in {time.time() - start_time:.2f} seconds")
            print_gpu_memory_stats(rank, "after dataset creation")
        
        # Create distributed sampler for training data
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank
        )
        
        if is_master:
            print(f"[Process {rank}] Creating data loaders with batch size {args.batch_size} and {args.num_workers} workers")
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=custom_collate,
            pin_memory=True,
            sampler=train_sampler
        )
        
        # Validation loader doesn't need to be distributed
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=custom_collate,
            pin_memory=True
        )
        
        if is_master:
            print(f"[Process {rank}] Train dataset size: {len(train_dataset)}")
            print(f"[Process {rank}] Validation dataset size: {len(valid_dataset)}")
            print(f"[Process {rank}] Train loader batches: {len(train_loader)}")
            print(f"[Process {rank}] Validation loader batches: {len(valid_loader)}")
            print_gpu_memory_stats(rank, "after dataloader creation")
        
        # Create models for the specified view
        models = {}
        optimizers = {}
        
        # Determine which views to create models for
        if args.view:
            # If view is specified, only create model for that view
            views_to_create = [args.view]
            if is_master:
                print(f"[Process {rank}] Creating model only for view: {args.view}")
        else:
            # If no view is specified, create models for all three views
            views_to_create = ['axial', 'coronal', 'sagittal']
            if is_master:
                print(f"[Process {rank}] Creating models for all views")
        
        # Create a model for each view
        for view in views_to_create:
            if is_master:
                print(f"[Process {rank}] Creating model for view: {view}")
            
            start_time = time.time()
            model = MRNetModel(backbone=args.backbone)
            if args.sync_bn:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            
            model = model.to(device)
            # Add find_unused_parameters=True to avoid DDP errors
            model = DDP(model, device_ids=[rank], find_unused_parameters=True)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            models[view] = model
            optimizers[view] = optimizer
            
            if is_master:
                print(f"[Process {rank}] Model for {view} created in {time.time() - start_time:.2f} seconds")
                print_gpu_memory_stats(rank, f"after {view} model creation")
        
        # Loss function
        criterion = torch.nn.BCEWithLogitsLoss()
        
        # Training loop
        best_auc = {model_name: 0.0 for model_name in models}
        
        if is_master:
            print("\n=== Training Configuration ===")
            print(f"Task: {args.task}")
            print(f"View: {args.view if args.view else 'all views'}")
            print(f"Training Approach: per_view")  # Hardcoded to per_view
            print(f"Backbone: {args.backbone}")
            print(f"Batch Size: {args.batch_size} per GPU")
            print(f"Total Batch Size: {args.batch_size * world_size}")
            print(f"Learning Rate: {args.lr}")
            print(f"Weight Decay: {args.weight_decay}")
            print(f"Number of Epochs: {args.epochs}")
            print(f"Number of GPUs: {world_size}")
            print(f"Number of Workers per GPU: {args.num_workers}")
            print(f"Data Augmentation: {'Enabled' if args.use_augmentation else 'Disabled'}")
            print(f"Max Slices per MRI: {max_slices}")
            print(f"Mixed Precision Training: {'Enabled' if use_amp else 'Disabled'}")
            print("==============================\n")
        
        # Synchronize processes before starting training
        dist.barrier()
        
        for epoch in range(args.epochs):
            if is_master:
                print(f"\n[Process {rank}] Starting epoch {epoch+1}/{args.epochs}")
                print_gpu_memory_stats(rank, f"start of epoch {epoch+1}")
            
            # Set epoch for sampler
            train_sampler.set_epoch(epoch)
            
            # Training phase
            for model_name, model in models.items():
                if is_master:
                    print(f"\n[Process {rank}] Training {model_name} model for epoch {epoch+1}")
                    epoch_start_time = time.time()
                
                model.train()
                running_loss = 0.0
                batch_count = 0
                processed_samples = 0
                
                for i, batch in enumerate(train_loader):
                    try:
                        # Ensure the required view is available
                        if model_name not in batch['available_views']:
                            if is_master and i % 20 == 0:
                                print(f"[Process {rank}] Skipping batch {i+1} - view {model_name} not available")
                            continue
                        
                        batch_start_time = time.time()
                        batch_size = batch['label'].size(0)
                        processed_samples += batch_size
                        
                        # Print detailed batch info for early batches
                        if is_master and i < 5:
                            print(f"\n=== Batch {i+1} Details ===")
                            print(f"Batch keys: {list(batch.keys())}")
                            print(f"Available views: {batch['available_views']}")
                            print(f"Label shape: {batch['label'].shape}")
                            for view in batch['available_views']:
                                print(f"{view} shape: {batch[view].shape}")
                                print(f"{view} memory: {batch[view].element_size() * batch[view].nelement() / (1024**2):.2f} MB")
                        
                        # Get labels and reshape to [batch_size, 1]
                        labels = batch['label'].to(device)
                        labels = labels.view(-1, 1)  # Reshape labels to match output
                        
                        # Track memory before forward pass
                        if is_master and i < 5:
                            print_gpu_memory_stats(rank, f"before forward pass (batch {i+1})")
                        
                        # Get the data for this view
                        data = batch[model_name].to(device)
                        
                        # Log shape information
                        if is_master and i % 10 == 0:
                            print(f"[Process {rank}] Processing {model_name} data with shape: {data.shape}")
                        
                        optimizer = optimizers[model_name]
                        optimizer.zero_grad()
                        
                        # Use mixed precision
                        with autocast('cuda'):
                            outputs = model(data)
                            loss = criterion(outputs, labels)
                        
                        # Use scaler for backward and step
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        
                        running_loss += loss.item()
                        batch_count += 1
                        batch_time = time.time() - batch_start_time
                        
                        # More frequent logging
                        if is_master and i % 5 == 0:
                            print(f"[Process {rank}] Epoch [{epoch+1}/{args.epochs}], Batch [{i+1}/{len(train_loader)}], "
                                  f"Loss: {loss.item():.4f}, Batch time: {batch_time:.2f}s, "
                                  f"Samples: {processed_samples}")
                            
                            if i > 0 and i % 20 == 0:
                                # Print memory usage every 20 batches
                                print_gpu_memory_stats(rank, f"during training - batch {i+1}")
                        
                        if writer and is_master and i % 10 == 0:
                            writer.add_scalar(f'{model_name}/train_loss', loss.item(), 
                                            epoch * len(train_loader) + i)
                        
                        # Track memory after backward pass
                        if is_master and i < 5:
                            print_gpu_memory_stats(rank, f"after backward pass (batch {i+1})")
                        
                        # Add after each batch completes
                        torch.cuda.empty_cache()
                        gc.collect()
                        
                    except Exception as e:
                        if is_master:
                            print(f"\n=== ERROR in batch {i+1} ===")
                            print(f"Error type: {type(e).__name__}")
                            print(f"Error message: {str(e)}")
                            print(f"Model: {model_name}")
                            print(f"Batch keys: {list(batch.keys()) if 'batch' in locals() else 'Unknown'}")
                            print(f"Available views: {batch['available_views'] if 'batch' in locals() and 'available_views' in batch else 'Unknown'}")
                            
                            # Print stack trace
                            import traceback
                            print("\nStack trace:")
                            traceback.print_exc()
                            
                            # Print memory state
                            print_gpu_memory_stats(rank, "at error")
                            
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
                
                # Print epoch summary
                if is_master and batch_count > 0:
                    avg_loss = running_loss / batch_count
                    epoch_time = time.time() - epoch_start_time
                    print(f"\n[Process {rank}] Epoch {epoch+1} - {model_name} Summary:")
                    print(f"Average Loss: {avg_loss:.4f}")
                    print(f"Total Time: {epoch_time:.2f}s")
                    print(f"Samples Processed: {processed_samples}")
                    print_gpu_memory_stats(rank, f"end of epoch {epoch+1} training")
            
            # Synchronize all processes before validation
            dist.barrier()
            
            # Validation phase (only on master process)
            if is_master:
                print(f"\n[Process {rank}] Starting validation for epoch {epoch+1}")
                with torch.no_grad():
                    for model_name, model in models.items():
                        model.eval()
                        val_loss = 0.0
                        val_pred = []
                        val_true = []
                        val_start_time = time.time()
                        
                        for j, batch in enumerate(valid_loader):
                            # Skip batches without required views
                            if model_name not in batch['available_views']:
                                continue
                            
                            # Get labels and reshape
                            labels = batch['label'].to(device)
                            labels = labels.view(-1, 1)
                            
                            data = batch[model_name].to(device)
                            
                            # Use mixed precision for validation too (just for inference)
                            if use_amp:
                                with autocast('cuda'):
                                    outputs = model(data)
                            else:
                                outputs = model(data)
                            
                            # Calculate loss
                            loss = criterion(outputs, labels)
                            val_loss += loss.item()
                            
                            # Store predictions and true labels
                            val_pred.extend(torch.sigmoid(outputs).cpu().numpy())
                            val_true.extend(labels.cpu().numpy())
                            
                            if (j + 1) % 5 == 0:
                                print(f"[Process {rank}] Validation batch: {j+1}/{len(valid_loader)}")
                        
                        # Calculate validation AUC
                        if val_true and val_pred:  # Only calculate if we have predictions
                            val_auc = roc_auc_score(val_true, val_pred)
                            avg_val_loss = val_loss / len(valid_loader) if len(valid_loader) > 0 else 0
                            val_time = time.time() - val_start_time
                            
                            print(f"\n[Process {rank}] Epoch {epoch+1} - {model_name} Validation Results:")
                            print(f"Validation Loss: {avg_val_loss:.4f}")
                            print(f"Validation AUC: {val_auc:.4f}")
                            print(f"Validation Time: {val_time:.2f}s")
                            
                            if writer:
                                writer.add_scalar(f'{model_name}/val_loss', avg_val_loss, epoch)
                                writer.add_scalar(f'{model_name}/val_auc', val_auc, epoch)
                            
                            # Save best model
                            if val_auc > best_auc[model_name]:
                                best_auc[model_name] = val_auc
                                # Save the module instead of DDP wrapper
                                save_path = os.path.join(args.output_dir, f'{model_name}_best.pth')
                                torch.save(model.module.state_dict(), save_path)
                                print(f"[Process {rank}] Saved new best model with validation AUC: {val_auc:.4f}")
                    
                    # Save checkpoint for this epoch
                    for model_name, model in models.items():
                        model_path = os.path.join(args.output_dir, f'{model_name}_epoch_{epoch}.pth')
                        # Save the module instead of DDP wrapper
                        torch.save(model.module.state_dict(), model_path)
                        print(f"[Process {rank}] Saved checkpoint for {model_name} at epoch {epoch+1}")
            
            # Synchronize all processes after validation
            dist.barrier()
        
        # Print final results (master process only)
        if is_master:
            print("\n[Process {rank}] Training completed!")
            print("Best validation AUC scores:")
            for model_name, auc in best_auc.items():
                print(f"{model_name}: {auc:.4f}")
            
            # Close tensorboard writer
            if writer:
                writer.close()
        
        # Clean up
        cleanup_ddp()
        
    except Exception as e:
        print(f"[Process {rank}] Error in training process: {str(e)}")
        import traceback
        print(traceback.format_exc())
        # Make sure to clean up even if there's an error
        if dist.is_initialized():
            cleanup_ddp()
        raise e

def print_system_memory():
    """Print CPU memory usage"""
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"CPU Memory Usage: {memory_info.rss / (1024 ** 3):.2f} GB")

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
    parser = argparse.ArgumentParser(description='Train MRNet models with Multi-GPU support')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Root directory of the project (default: project root)')
    parser.add_argument('--task', type=str, default='abnormal',
                        choices=['abnormal', 'acl', 'meniscus'],
                        help='Task to train on')
    parser.add_argument('--view', type=str, default=None,
                        choices=['axial', 'coronal', 'sagittal'],
                        help='MRI view to train on (for single-view training)')
    
    # Model parameters
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['alexnet', 'resnet18', 'densenet121'],
                        help='Backbone architecture')
    parser.add_argument('--train_approach', type=str, default='per_view',
                        choices=['per_view'],  # Only per_view is available now
                        help='Training approach')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,  # Reduced default from 16 to 8
                        help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=1,  # Reduced default from 4 to 1
                        help='Number of workers for data loading')
    parser.add_argument('--use_augmentation', action='store_true',
                        help='Enable data augmentation during training')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='model_outputs',
                        help='Directory to save models and logs')
    
    # Multi-GPU specific parameters
    parser.add_argument('--master_port', type=str, default='12355',
                        help='Port for distributed training')
    parser.add_argument('--sync_bn', action='store_true',
                        help='Use synchronized batch normalization')
    parser.add_argument('--scale_lr', action='store_true',
                        help='Scale learning rate by number of GPUs')
    
    # Add a new argument to control GPU usage
    parser.add_argument('--num_gpus', type=int, default=None,
                        help='Number of GPUs to use (default: all available minus one)')
    parser.add_argument('--use_all_gpus', action='store_true',
                        help='Use all available GPUs instead of reserving one')
    
    args = parser.parse_args()
    
    # Override train_approach to always be per_view
    args.train_approach = 'per_view'
    
    # Get number of available GPUs
    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        raise ValueError("No GPUs are available. Multi-GPU training cannot proceed.")
    
    # Decide how many GPUs to use
    if args.num_gpus is not None:
        # User explicitly specified number of GPUs
        num_gpus = min(args.num_gpus, available_gpus)
    else:
        # By default, use all available GPUs minus one (unless use_all_gpus is True)
        if args.use_all_gpus:
            num_gpus = available_gpus
        else:
            num_gpus = max(1, available_gpus - 1)  # Use at least 1 GPU
    
    # Scale learning rate if requested
    if args.scale_lr:
        args.lr = args.lr * num_gpus
        print(f"Scaling learning rate to {args.lr} for {num_gpus} GPUs")
    
    # Set data directory to project root if not specified
    if args.data_dir is None:
        args.data_dir = get_project_root()
    
    print(f"Starting distributed training on {num_gpus} GPUs (out of {available_gpus} available)")
    
    # Launch distributed training
    mp.spawn(
        train_model_ddp,
        args=(num_gpus, args),
        nprocs=num_gpus,
        join=True
    )

if __name__ == "__main__":
    main()
