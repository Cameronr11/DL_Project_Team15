import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
import sys
import os
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Now import using the correct path
from src.data_loader import MRNetDataset
import time
import glob

def get_project_root():
    """Returns the absolute path to the project root directory"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def check_data_structure():
    """Check the structure of data directories"""
    project_root = get_project_root()
    
    print("\n=== Data Structure Check ===")
    
    # List of potential processed data locations
    potential_paths = [
        os.path.join(project_root, 'processed_data'),
        os.path.join(project_root, 'src', 'processed', 'data'),
        os.path.join(project_root, 'src', 'processed_data'),
        os.path.join(project_root, 'processed'),
        os.path.join(project_root, 'src', 'processed')
    ]
    
    found_processed_data = False
    
    for path in potential_paths:
        if os.path.exists(path):
            print(f"Found potential processed data directory: {path}")
            
            # Check for view subdirectories
            has_views = True
            for view in ['axial', 'coronal', 'sagittal']:
                view_dir = os.path.join(path, view)
                if os.path.exists(view_dir):
                    # Count files
                    npy_files = glob.glob(os.path.join(view_dir, "*.npy"))
                    print(f"Found {len(npy_files)} .npy files in {view} directory")
                    if npy_files:
                        # Show a few example filenames
                        examples = [os.path.basename(f) for f in npy_files[:5]]
                        print(f"Example filenames: {', '.join(examples)}")
                else:
                    print(f"WARNING: {view} directory not found at {view_dir}")
                    has_views = False
            
            if has_views:
                print(f"CONFIRMED: {path} appears to be a valid processed data directory with view subdirectories")
                found_processed_data = True
            else:
                print(f"NOTE: {path} exists but does not contain all view subdirectories")
    
    if not found_processed_data:
        print("WARNING: Could not find a complete processed data directory with axial, coronal, and sagittal subdirectories")
        
        # Look for alternative directories containing MRI data
        print("Searching for .npy files in project directory...")
        found_files = []
        for root, dirs, files in os.walk(project_root):
            # Skip large directories that might slow down the search
            if '.git' in root or 'venv' in root or '__pycache__' in root:
                continue
                
            for file in files:
                if file.endswith('.npy'):
                    found_files.append(os.path.join(root, file))
                    if len(found_files) <= 10:  # Show first 10 files
                        print(f"Found: {os.path.join(root, file)}")
        
        if found_files:
            print(f"Found {len(found_files)} .npy files in total")
            
            # Try to figure out the directory structure
            directories = set(os.path.dirname(f) for f in found_files)
            print(f"Files are located in {len(directories)} different directories")
            for directory in directories:
                file_count = len([f for f in found_files if os.path.dirname(f) == directory])
                print(f"- {directory}: {file_count} files")
            
            common_parent = os.path.commonpath(directories) if directories else None
            if common_parent:
                print(f"Common parent directory: {common_parent}")
        else:
            print("No .npy files found in the project directory")
    
    # Check original data location
    mrnet_dir = os.path.join(project_root, 'data', 'MRNet-v1.0')
    if os.path.exists(mrnet_dir):
        print(f"Found MRNet-v1.0 directory: {mrnet_dir}")
        
        # Check for train and valid CSV files
        csv_files = glob.glob(os.path.join(mrnet_dir, "*.csv"))
        print(f"Found {len(csv_files)} CSV files in MRNet-v1.0 directory")
        if csv_files:
            examples = [os.path.basename(f) for f in csv_files]
            print(f"CSV files: {', '.join(examples)}")

def custom_collate(batch):
    """
    Custom collate function that handles batches with varying available views.
    Only include views that are available in all samples in the batch.
    """
    # Find common views available in all samples
    available_in_all = {'axial', 'coronal', 'sagittal'}
    for sample in batch:
        available_in_sample = set(sample['available_views'])
        available_in_all = available_in_all.intersection(available_in_sample)
    
    # If no common views, create empty batch with just labels and case_ids
    if not available_in_all:
        labels = torch.stack([sample['label'] for sample in batch])
        case_ids = [sample['case_id'] for sample in batch]
        return {'label': labels, 'case_id': case_ids, 'available_views': []}
    
    # Create batch with only common views
    result = {
        'label': torch.stack([sample['label'] for sample in batch]),
        'case_id': [sample['case_id'] for sample in batch],
        'available_views': list(available_in_all)
    }
    
    # Add tensors for common views
    for view in available_in_all:
        result[view] = torch.stack([sample[view] for sample in batch])
    
    return result

def test_mrnet_dataset():
    # First check data directory structure
    check_data_structure()
    
    # Get absolute path to project root
    project_root = get_project_root()
    
    # Initialize dataset with correct root path
    dataset = MRNetDataset(
        root_dir=project_root,
        task='abnormal',
        train=True
    )
    
    # Basic checks
    print(f"\nDataset size: {len(dataset)}")
    
    # Check first few samples individually to debug
    print("\n=== Checking Individual Samples ===")
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        print(f"Sample {i}: Case ID {sample['case_id']}, Label {sample['label'].item()}")
        print(f"  Available views: {sample['available_views']}")
        for view in sample['available_views']:
            print(f"  {view} shape: {sample[view].shape}")
    
    if not any(len(dataset[i]['available_views']) > 0 for i in range(min(20, len(dataset)))):
        print("\nWARNING: No views found for any samples! Data may not be properly preprocessed or accessible.")
        return
    
    # Continue with rest of tests only if we have some data
    # Test DataLoader with custom collate function
    dataloader = DataLoader(
        dataset, 
        batch_size=4,  # Use a smaller batch size for testing 
        shuffle=True, 
        num_workers=1,  # Use fewer workers for testing
        collate_fn=custom_collate
    )
    
    # Try to get a single batch
    print("\n=== Testing DataLoader ===")
    try:
        batch = next(iter(dataloader))
        print(f"Successfully loaded a batch with {len(batch['case_id'])} samples")
        print(f"Available views in batch: {batch.get('available_views', [])}")
        for view in batch.get('available_views', []):
            print(f"  {view} batch shape: {batch[view].shape}")
    except Exception as e:
        print(f"Error loading batch: {e}")

if __name__ == "__main__":
    test_mrnet_dataset()