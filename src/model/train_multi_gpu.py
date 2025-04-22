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

# This is importing the components we have created
from src.data_loader import MRNetDataset, SimpleMRIAugmentation
from src.model.MRNetModel import MRNetModel
from src.utils.metric_tracker import MetricTracker

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
            print(f"Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    else:
        print("WARNING: CUDA is not available. Multi-GPU training cannot proceed.")
    print("=====================\n")



#allows us to track gpu memory usage
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




#This is for the distributed training on GPU's currently we can only train on one GPU at a time
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



#cleans up the distributed processes
def cleanup_ddp():
    """
    Cleanup distributed processes
    """
    dist.destroy_process_group()



#main training function
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
        #this needs to be checked I don't know if this is correct but it does affect the efficiency of the model
        use_amp = True
        scaler = GradScaler() if use_amp else None
        
        # Only print from master process
        is_master = rank == 0
        



        # Initialize TensorBoard writer (only on master process) There are alternatives to tensorboard that we need to evaluate
        writer = None
        if is_master:
            log_dir = os.path.join(args.output_dir, 'logs')
            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=log_dir)
            print(f"[Process {rank}] TensorBoard logs will be saved to {log_dir}")
        



        # Initialize MetricTracker that we created to track the metrics of the model
        metric_tracker = None
        if is_master:
            model_name = f"{args.backbone}_{args.task}_{args.view}"
            config = vars(args)  # Convert args to dictionary
            metric_tracker = MetricTracker(
                model_name=model_name,
                task=args.task,
                view=args.view,
                config=config,
                output_dir=args.output_dir
            )
            print(f"[Process {rank}] Metrics will be tracked and saved to {args.output_dir}")
        

        #this is for printing the initial training process and printed the intial gpu memory stats
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
        

        #all this is for printing the configuration of the training process to keep track of the arguments for a specific run
        if is_master:
            print(f"\n=== Configuration ===")
            print(f"Project root: {project_root}")
            print(f"Output directory: {args.output_dir}")
            print(f"Task: {args.task}")
            print(f"View: {args.view}")
            print(f"Training approach: {args.train_approach}")
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
        


        #handles the slice count for each sample. This can be evaluated to see what the max number of slices we can use
        #ideally we should want the max number of slices for each sample to be used that doesn't mess with the memory usage
        start_time = time.time()
        # Determine which view to load if using per_view approach
        view_to_load = args.view
        
        # Determine max slices based on backbone model
        if args.max_slices is not None:
            max_slices = args.max_slices  # Use user-specified value
        elif args.backbone == 'densenet121':
            max_slices = 32  # Use fewer slices for DenseNet121
        else:  # resnet18 or others
            max_slices = 64  # Use more slices for other backbones
            
        if is_master:
            print(f"[Process {rank}] Using {max_slices} max slices for {args.backbone}")
        


        #setting up datasets and the loaders for those datasets
        # Create datasets with updated parameters
        root = args.data_dir or get_project_root()
        train_dataset = MRNetDataset(
            root_dir=root,
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
            root_dir=root,
            task=args.task,
            split='valid',
            transform=None,
            max_slices=max_slices,
            view=view_to_load  # Pass the view to load
        )
        if is_master:
            print(f"[Process {rank}] Validation dataset created in {time.time() - start_time:.2f} seconds")
            print_gpu_memory_stats(rank, "after dataset creation")
        
        # Create distributed sampler for training data (Figure out what this is doing)
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
        schedulers = {}  # Add schedulers dictionary
        
        # Determine which views to create models for
        if args.view:
            # If view is specified, only create model for that view
            views_to_create = [args.view]
            if is_master:
                print(f"[Process {rank}] Creating model only for view: {args.view}")
        else:
            # If no view is specified, create models for all three views, this should not be necessary as we will always be specifying a view
            views_to_create = ['axial', 'coronal', 'sagittal']
            if is_master:
                print(f"[Process {rank}] Creating models for all views")
        
        # Create a model for each view, need to simplify this for just one view
        for view in views_to_create:
            if is_master:
                print(f"[Process {rank}] Creating model for view: {view}") 
            start_time = time.time()
            model = MRNetModel(backbone=args.backbone)
            if args.sync_bn:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            
            model = model.to(device)
            # Add find_unused_parameters=True to avoid DDP errors
            model = DDP(model, device_ids=[rank])

            #this is our optimizer, we can try different optimizers and see what works best
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            # Create learning rate scheduler, this is a simple scheduler that reduces the learning rate by 50% if the validation loss doesn't improve after 5 epochs
            #we can evaluate different schedulers and see what works best
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=is_master
            )
            

            #this is for storing the models, optimizers, and schedulers for each view
            models[view] = model
            optimizers[view] = optimizer
            schedulers[view] = scheduler  # Store the scheduler
            




            if is_master:
                print(f"[Process {rank}] Model for {view} created in {time.time() - start_time:.2f} seconds")
                print_gpu_memory_stats(rank, f"after {view} model creation")
        
        # Loss function, determine if this is the best loss function for our model
        criterion = torch.nn.BCEWithLogitsLoss()
        
        # Training loop
        best_val_auc = 0.0
        best_model_path = os.path.join(args.output_dir, f"best_model_{args.backbone}_{args.task}_{args.view}.pth")
        global_step = 0
        
        if is_master:
            print("\n=== Training Configuration ===")
            print(f"Task: {args.task}")
            print(f"View: {args.view if args.view else 'all views'}")
            print(f"Training Approach: {args.train_approach}")
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
        
        #main training loop
        for epoch in range(args.epochs):
            if is_master:
                print(f"\n[Process {rank}] Starting epoch {epoch+1}/{args.epochs}")
                print_gpu_memory_stats(rank, f"start of epoch {epoch+1}")
            
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
                        if is_master and batch_idx % 20 == 0:
                            print(f"[Process {rank}] Skipping batch {batch_idx+1} - view {args.view} not available")
                        continue
                    
                    batch_start_time = time.time()
                    batch_size = batch['label'].size(0)
                    
                    # Print detailed batch info for early batches
                    if is_master and batch_idx < 5:
                        print(f"\n=== Batch {batch_idx+1} Details ===")
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
                    if is_master and batch_idx < 5:
                        print_gpu_memory_stats(rank, f"before forward pass (batch {batch_idx+1})")
                    
                    # Get the data for this view
                    data = batch[args.view].to(device)
                    
                    # Log shape information
                    if is_master and batch_idx % 10 == 0:
                        print(f"[Process {rank}] Processing {args.view} data with shape: {data.shape}")
                    
                    optimizer = optimizers[args.view]
                    optimizer.zero_grad() #this is for zeroing out the gradients
                    
                    # Forward pass with automatic mixed precision
                    with autocast(device_type='cuda', enabled=use_amp):
                        outputs = model(data)
                    
                    # Calculate loss
                    loss = criterion(outputs, labels)
                    

                    # massive piece that we need to evaluate 
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
                    log_interval = getattr(args, 'log_interval', 10)  # Default to 10 if not set
                    if is_master and batch_idx % log_interval == 0:
                        avg_loss = running_loss / (batch_idx + 1)
                        
                        # Calculate training AUC if we have enough samples
                        train_auc = 0.0
                        if len(train_true) > 1 and len(np.unique(train_true)) > 1:
                            train_auc = roc_auc_score(train_true, train_pred)
                        
                        # Log to TensorBoard, I dont know if tensorboard is the best tool for this job
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
                        print(f"[Process {rank}] Epoch {epoch}/{args.epochs} | Batch {batch_idx}/{len(train_loader)} | "
                              f"Loss: {avg_loss:.4f}" + (f" | AUC: {train_auc:.4f}" if train_auc > 0 else ""))
                    
                    global_step += 1 #why is this needed?
                    
                    # Track memory after backward pass
                    if is_master and batch_idx < 5:
                        print_gpu_memory_stats(rank, f"after backward pass (batch {batch_idx+1})")
                    
                    # Add after each batch completes
                    if args.debug:
                        torch.cuda.empty_cache()
                        gc.collect()
                    
                except Exception as e:
                    if is_master:
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
            
            # End of epoch - calculate final training metrics
            train_loss = running_loss / len(train_loader)
            train_auc = roc_auc_score(train_true, train_pred) if len(np.unique(train_true)) > 1 else 0.0
            
            # Evaluate on validation set
            val_loss, val_auc, val_true, val_pred = evaluate(model, device, valid_loader, criterion, args)
            
            # Log validation metrics
            if is_master:
                print(f"\n[Process {rank}] Epoch {epoch}/{args.epochs} completed | "
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
                    torch.save(model.module.state_dict(), best_model_path)
                    print(f"[Process {rank}] New best model saved with Val AUC: {val_auc:.4f}")
                
                # Save epoch predictions for later analysis (only for best epoch)
                if val_auc == best_val_auc and metric_tracker:
                    np.save(os.path.join(args.output_dir, 'val_true.npy'), np.array(val_true))
                    np.save(os.path.join(args.output_dir, 'val_pred.npy'), np.array(val_pred))
                
                # Adjust learning rate using the scheduler
                scheduler = schedulers[args.view]
                scheduler.step(val_loss)
            
            # Synchronize processes to ensure all are ready for next epoch
            dist.barrier()
        
        # End of training - cleanup and save final metrics
        if is_master:
            print(f"\n[Process {rank}] Training completed | Best Val AUC: {best_val_auc:.4f}")
            
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
                    print(f"[Process {rank}] Training progress report saved to {args.output_dir}")
                except Exception as e:
                    print(f"[Process {rank}] Error generating training report: {str(e)}")
            
            # Close TensorBoard writer
            if writer:
                writer.close()
        
        # Cleanup DDP
        cleanup_ddp()
        
        return best_val_auc
    
    except Exception as e:
        print(f"Error in process {rank}: {str(e)}")
        import traceback
        traceback.print_exc()
        cleanup_ddp()
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

def print_system_memory():
    """Print CPU memory usage"""
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"CPU Memory Usage: {memory_info.rss / (1024 ** 3):.2f} GB")


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
                        choices=['alexnet', 'resnet18', 'resnet34', 'densenet121'],
                        help='Backbone architecture')
    parser.add_argument('--train_approach', type=str, default='per_view',
                        choices=['per_view'],  # Only per_view is available now
                        help='Training approach')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size per GPU')
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
    
    # Add arguments for metrics tracking and visualization
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save predictions from validation and test sets for later analysis')
    parser.add_argument('--generate_plots', action='store_true',
                       help='Generate plots of training progress and model performance')
    
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
