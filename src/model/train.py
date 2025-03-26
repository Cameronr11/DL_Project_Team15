import sys
import os

# Add the project root directory to Python path, this makes sure you can run it from any directory
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import roc_auc_score
import argparse
import time
import matplotlib.pyplot as plt
import pandas as pd
import psutil  # For detecting number of CPU cores

from src.data_loader import MRNetDataset
from src.model.MRNetModel import MRNetModel, MRNetEnsemble
from src.data_loader import SimpleMRIAugmentation

def get_project_root():
    """Returns the absolute path to the project root directory"""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#for if we want to try training on a GPU
def verify_gpu():
    """Verify GPU availability and print information"""
    print("\n=== GPU Information ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    else:
        print("WARNING: CUDA is not available. Training will proceed on CPU.")
        print("This will be significantly slower than GPU training.")
    print("=====================\n")

def setup_ddp(rank, world_size, args):
    """
    Setup for Distributed Data Parallel on CPU
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.master_port
    # Use 'gloo' backend for CPU training instead of 'nccl'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    # Set this process's device
    torch.set_num_threads(1)  # Important to avoid oversubscription

def cleanup_ddp():
    """
    Clean up process group
    """
    dist.destroy_process_group()

def train_model_ddp(rank, world_size, args):
    """
    Training function for distributed training on CPU
    """
    # Setup DDP
    setup_ddp(rank, world_size, args)
    
    # For CPU training, we use CPU device
    device = torch.device('cpu')
    
    # Only print from master process
    is_master = rank == 0
    
    if is_master:
        print(f"Starting distributed training with {world_size} processes...", flush=True)
        verify_gpu()
        print(f"Setting up device and paths...", flush=True)
        print(f"Using device: {device} for rank {rank}", flush=True)
    
    project_root = get_project_root()
    if is_master:
        print(f"Project root: {project_root}", flush=True)
        print(f"Creating output directory: {args.output_dir}", flush=True)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup tensorboard for master process only
    writer = None
    if is_master:
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'))
        print("\n=== Checking Data Directory Structure ===", flush=True)
        
        data_dir = os.path.join(project_root, 'data', 'MRNet-v1.0')
        processed_dir = os.path.join(data_dir, 'processed_train_data')
        print(f"Looking for data in: {processed_dir}", flush=True)
        
        # Check train directories for each view
        for view in ['axial', 'coronal', 'sagittal']:
            train_dir = os.path.join(processed_dir, f"{view}_train")
            test_dir = os.path.join(processed_dir, f"{view}_test")
            valid_dir = os.path.join(processed_dir, view)
            
            print(f"\nChecking {view} directories:")
            if os.path.exists(train_dir):
                print(f"Train directory exists: {train_dir}")
                print("Sample contents:", os.listdir(train_dir)[:5])
            else:
                print(f"ERROR: Train directory not found: {train_dir}")
            
            if os.path.exists(test_dir):
                print(f"Test directory exists: {test_dir}")
                print("Sample contents:", os.listdir(test_dir)[:5])
            else:
                print(f"ERROR: Test directory not found: {test_dir}")
            
            if os.path.exists(valid_dir):
                print(f"Valid directory exists: {valid_dir}")
                print("Sample contents:", os.listdir(valid_dir)[:5])
            else:
                print(f"ERROR: Valid directory not found: {valid_dir}")
    
    # Synchronize processes after initialization
    dist.barrier()
    
    if is_master:
        print("\nInitializing datasets...", flush=True)
        print(f"Creating train dataset for task: {args.task}", flush=True)
    
    train_dataset = MRNetDataset(
        root_dir=project_root,
        task=args.task,
        split='train',
        transform=SimpleMRIAugmentation(p=0.5) if args.use_augmentation else None
    )
    if is_master:
        print("Train dataset created successfully", flush=True)
        print("Creating validation dataset...", flush=True)
    
    valid_dataset = MRNetDataset(
        root_dir=project_root,
        task=args.task,
        split='valid',
        transform=None
    )
    if is_master:
        print("Validation dataset created successfully", flush=True)
        print("Creating test dataset...", flush=True)
    
    test_dataset = MRNetDataset(
        root_dir=project_root,
        task=args.task,
        split='test',
        transform=None
    )
    if is_master:
        print("Test dataset created successfully", flush=True)
    
    # Create distributed sampler for training data
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank
    )
    
    if is_master:
        print("\nInitializing data loaders...", flush=True)
        print(f"Creating train loader with batch size: {args.batch_size}", flush=True)
    
    # Create distributed data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Sampler handles shuffling
        num_workers=0,  # Use 0 to avoid spawning more subprocesses
        collate_fn=MRNetDataset.custom_collate,
        sampler=train_sampler,
        pin_memory=False,
        persistent_workers=False
    )
    
    # Validation and test loaders don't need to be distributed
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=MRNetDataset.custom_collate,
        pin_memory=False,
        persistent_workers=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=MRNetDataset.custom_collate,
        pin_memory=False,
        persistent_workers=False
    )
    
    if is_master:
        print("Data loaders created successfully", flush=True)
    
    # Create model based on training approach
    if args.train_approach == 'per_view':
        # Train a separate model for each view
        if is_master:
            print(f"Setting up per-view training with backbone: {args.backbone}", flush=True)
        
        models = {}
        optimizers = {}
        
        for view in ['axial', 'coronal', 'sagittal']:
            if is_master:
                print(f"Creating model for view: {view}", flush=True)
            
            model = MRNetModel(backbone=args.backbone)
            model = model.to(device)
            # Wrap model with DDP (but with device_ids=None for CPU)
            model = DDP(model, device_ids=None, find_unused_parameters=True)
            
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            models[view] = model
            optimizers[view] = optimizer
            
            if is_master:
                print(f"Model for {view} created successfully", flush=True)
        
        # Only train the specified view if provided
        if args.view:
            if args.view not in models:
                raise ValueError(f"Invalid view: {args.view}. Choose from 'axial', 'coronal', 'sagittal'")
            models = {args.view: models[args.view]}
            optimizers = {args.view: optimizers[args.view]}
           
    elif args.train_approach == 'ensemble':
        # Train ensemble model
        model = MRNetEnsemble(backbone=args.backbone)
        model = model.to(device)
        model = DDP(model, device_ids=None)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        models = {'ensemble': model}
        optimizers = {'ensemble': optimizer}
    else:
        raise ValueError(f"Invalid training approach: {args.train_approach}")
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    best_auc = {model_name: 0.0 for model_name in models}
    best_val_auc = 0.0  # Track best validation performance
    
    if is_master:
        print("\n=== Training Configuration ===")
        print(f"Task: {args.task}")
        print(f"View: {args.view}")
        print(f"Training Approach: {args.train_approach}")
        print(f"Batch Size: {args.batch_size} per process")
        print(f"Total Batch Size: {args.batch_size * world_size}")
        print(f"Learning Rate: {args.lr}")
        print(f"Number of Epochs: {args.epochs}")
        print(f"Device: {device}")
        print(f"Number of Processes: {world_size}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(valid_dataset)}\n")

    for epoch in range(args.epochs):
        if is_master:
            print(f"\nStarting epoch {epoch+1}/{args.epochs}", flush=True)
        
        # Set epoch for sampler to ensure proper shuffling
        train_sampler.set_epoch(epoch)
        
        for model_name, model in models.items():
            if is_master:
                print(f"\nTraining {model_name} model for epoch {epoch+1}", flush=True)
            
            model.train()
            running_loss = 0.0
            batch_count = 0
            
            if is_master:
                print(f"\nTraining {model_name} model:")
                print(f"Total batches: {len(train_loader)}")
            
            for batch_idx, batch in enumerate(train_loader):
                try:
                    if model_name not in batch['available_views']:
                        if is_master and batch_idx % 10 == 0:  # Only show every 10th skip to reduce output
                            print(f"Skipping batch {batch_idx+1}/{len(train_loader)} - missing view {model_name}", flush=True)
                        continue
                    
                    if is_master and batch_idx % 5 == 0:
                        print(f"Processing batch {batch_idx+1}/{len(train_loader)} for {model_name}", flush=True)
                    
                    labels = batch['label'].to(device)
                    data = batch[model_name].to(device)
                    
                    optimizer = optimizers[model_name]
                    optimizer.zero_grad()
                    outputs = model(data)
                    
                    # Inside your training loop, before calculating the loss
                    labels = labels.view(-1, 1)  # Reshape from [batch_size] to [batch_size, 1]
                    loss = criterion(outputs, labels)
                    
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    batch_count += 1
                    
                    # Show progress every 5 batches (master process only)
                    if is_master and (batch_idx + 1) % 5 == 0:
                        avg_loss = running_loss / batch_count
                        print(f"Batch [{batch_idx+1}/{len(train_loader)}] - Loss: {avg_loss:.4f}")
                    
                except Exception as e:
                    if is_master:
                        print(f"Error in batch {batch_idx+1}: {str(e)}", flush=True)
                        import traceback
                        print(traceback.format_exc(), flush=True)
                    raise e
            
            # Print epoch summary (master process only)
            if is_master and batch_count > 0:
                epoch_loss = running_loss / batch_count
                print(f"\nEpoch {epoch+1} Training Summary:")
                print(f"Average Loss: {epoch_loss:.4f}")
            
            # Synchronize before validation
            dist.barrier()
            
            # Validation phase (only on master process to avoid duplicate work)
            if is_master:
                # Validation phase
                with torch.no_grad():
                    for model_name, model in models.items():
                        model.eval()
                        val_loss = 0.0
                        val_pred = []
                        val_true = []
                        
                        for batch_idx, batch in enumerate(valid_loader):
                            # Skip batches without required views
                            if args.train_approach == 'per_view' and model_name not in batch['available_views']:
                                continue
                            
                            # Get labels
                            labels = batch['label'].to(device)
                            # Reshape labels to match output shape [batch_size, 1]
                            labels = labels.view(-1, 1)
                            
                            if args.train_approach == 'per_view':
                                # For per-view validation, use just that view
                                if model_name not in batch['available_views']:
                                    continue
                                
                                data = batch[model_name].to(device)
                                outputs = model(data)
                                
                            elif args.train_approach == 'ensemble':
                                # For ensemble validation, pass all views
                                data_dict = {view: batch[view].to(device) for view in batch['available_views']}
                                
                                if not data_dict:  # Skip if no views available
                                    continue
                                
                                outputs = model(data_dict)
                            
                            # Calculate loss
                            loss = criterion(outputs, labels)
                            val_loss += loss.item()
                            
                            # Store predictions and true labels
                            val_pred.extend(torch.sigmoid(outputs).cpu().numpy())
                            val_true.extend(labels.cpu().numpy())
                            
                            if (batch_idx + 1) % 5 == 0:
                                print(f"Processed validation batch: {batch_idx+1}/{len(valid_loader)}")
                        
                        # Calculate validation AUC
                        if val_true and val_pred:  # Only calculate if we have predictions
                            val_auc = roc_auc_score(val_true, val_pred)
                            avg_val_loss = val_loss / len(valid_loader)
                            
                            print(f"\nEpoch {epoch+1} Validation Results:")
                            print(f"Validation Loss: {avg_val_loss:.4f}")
                            print(f"Validation AUC: {val_auc:.4f}")
                            
                            # Save best model
                            if val_auc > best_auc[model_name]:
                                best_auc[model_name] = val_auc
                                save_path = os.path.join(args.output_dir, f'{model_name}_best.pth')
                                # Save the module instead of DDP wrapper
                                torch.save(model.module.state_dict(), save_path)
                                print(f"Saved new best model with validation AUC: {val_auc:.4f}")
                
                # Save checkpoint for each epoch
                for model_name, model in models.items():
                    model_path = os.path.join(args.output_dir, f'{model_name}_epoch_{epoch}.pth')
                    # Save the module instead of DDP wrapper
                    torch.save(model.module.state_dict(), model_path)
                
                # In your training loop, after validation
                if val_auc > best_val_auc:
                    print("\nEvaluating on test set...")
                    model.eval()
                    test_pred = []
                    test_true = []
                    
                    with torch.no_grad():
                        for batch in test_loader:
                            # Your existing evaluation code
                            labels = batch['label'].to(device)
                            if args.train_approach == 'per_view':
                                if model_name not in batch['available_views']:
                                    continue
                                data = batch[model_name].to(device)
                                outputs = model(data)
                            else:  # ensemble
                                data_dict = {view: batch[view].to(device) for view in batch['available_views']}
                                outputs = model(data_dict)
                            
                            test_pred.extend(torch.sigmoid(outputs).cpu().numpy())
                            test_true.extend(labels.cpu().numpy())
                    
                    if test_true and test_pred:
                        test_auc = roc_auc_score(test_true, test_pred)
                        print(f'Test AUC: {test_auc:.3f}')
                        writer.add_scalar(f'{model_name}/test_auc', test_auc, epoch)
            
            # Synchronize after validation
            dist.barrier()
    
    # Print final results (master process only)
    if is_master:
        print("\nTraining completed!")
        print("Best validation AUC scores:")
        for model_name, auc in best_auc.items():
            print(f"{model_name}: {auc:.4f}")
        
        # Close tensorboard writer
        if writer:
            writer.close()
    
    # Clean up
    cleanup_ddp()

def train_model(args):
    """
    Launch distributed training across multiple CPU cores
    """
    # Determine number of processes to use
    if args.num_processes is None:
        # Use all logical cores except one (to keep system responsive)
        args.num_processes = max(1, psutil.cpu_count(logical=True) - 1)
    
    # Scale learning rate if requested
    if args.scale_lr:
        args.lr = args.lr * args.num_processes
        print(f"Scaling learning rate to {args.lr} for {args.num_processes} processes")
    
    print(f"Starting distributed training on {args.num_processes} CPU processes")
    
    # Launch distributed training
    mp.spawn(
        train_model_ddp,
        args=(args.num_processes, args),
        nprocs=args.num_processes,
        join=True
    )

def custom_collate(batch):
    """
    Custom collate function for DataLoader
    Args:
        batch: List of samples from __getitem__
    Returns:
        Properly collated batch
    """
    try:
        # Create result dict with all samples
        result = {
            'label': torch.stack([sample['label'] for sample in batch]),
            'case_id': [sample['case_id'] for sample in batch],
            'available_views': []  # Will be populated with common views
        }
        
        # Check which views are available in all samples
        for view in ['axial', 'coronal', 'sagittal']:
            # Check if view exists in all samples
            if all(view in sample['available_views'] for sample in batch):
                tensors = [sample[view] for sample in batch]
                result[view] = torch.stack(tensors)
                result['available_views'].append(view)
        
        return result
        
    except Exception as e:
        print(f"Error in collate function: {str(e)}", flush=True)
        import traceback
        print(traceback.format_exc(), flush=True)
        raise e

#allows us to run the script from the command line and set all the appropriate parameters

# To run use something like this: python train.py --data_dir "path/to/data" --task "abnormal" --view "axial" --backbone "alexnet" --train_approach "per_view" --batch_size 8 --epochs 50 --lr 1e-5 --weight_decay 1e-4 --num_workers 4 --output_dir "model_outputs" --no_cuda --gpu 0
#Can run without some of these flags
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MRNet models')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Root directory of the project (default: project root)')
    parser.add_argument('--task', type=str, default='abnormal',
                        choices=['abnormal', 'acl', 'meniscus'],
                        help='Task to train on')
    parser.add_argument('--view', type=str, default='axial',
                        choices=['axial', 'coronal', 'sagittal'],
                        help='MRI view to train on (for single-view training)')
    
    # Model parameters
    parser.add_argument('--backbone', type=str, default='alexnet',
                        choices=['alexnet', 'resnet18', 'densenet121'],
<<<<<<< HEAD
                        help='Neural network backbone to use')
=======
                        help='Backbone architecture')
>>>>>>> e1c80908228f173060b9a8f591340f1f5bd63a3f
    parser.add_argument('--train_approach', type=str, default='per_view',
                        choices=['per_view', 'ensemble'],
                        help='Training approach')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training (per process)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='model_outputs',
                        help='Directory to save models and logs')
    
    # Hardware parameters
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA training')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    
    # Multi-CPU specific parameters
    parser.add_argument('--num_processes', type=int, default=None,
                        help='Number of processes to use (default: num_cores - 1)')
    parser.add_argument('--master_port', type=str, default='12355',
                        help='Port for distributed training')
    parser.add_argument('--scale_lr', action='store_true',
                        help='Scale learning rate by number of processes')
    
    # Other parameters
    parser.add_argument('--use_augmentation', action='store_true',
                        help='Enable data augmentation during training')
    
    args = parser.parse_args()
    
    # Set data directory to project root if not specified
    if args.data_dir is None:
        args.data_dir = get_project_root()
    
    train_model(args)