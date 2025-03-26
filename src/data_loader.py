"""
MRNet Dataset Loader

This module handles loading and preprocessing of MRI knee scan data for the MRNet project.
It provides a PyTorch Dataset for working with single-view MRI data.

Key components:
- MRNetDataset: PyTorch Dataset for loading MRI data from a specific view (axial, coronal, or sagittal)
- Custom collation function: For handling variable-sized MRI volumes in batches
- Utility functions: For easily creating properly configured data loaders
"""

import torch
import pandas as pd
import numpy as np
import os
import glob
from torch.utils.data import Dataset, DataLoader
from src.data_augmentation import SimpleMRIAugmentation
from sklearn.model_selection import train_test_split
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


#IF this is not setup correctly the entire model will fail, so it is important to get this right
#We need to go through this whole file and ensure that the data is loaded in the correct format
#This is the main file that will be used to load the data for the model




class MRNetDataset(torch.utils.data.Dataset):
    """
    Dataset for MRNet knee MRI data with single view approach.
    
    This dataset handles loading of MRI data from a specific anatomical view:
    - Axial (top-down view)
    - Coronal (front-to-back view)
    - Sagittal (side view)
    
    Each MRI consists of multiple 2D slices that together form a 3D volume.
    """
    def __init__(self, root_dir, task, split='train', transform=None, max_slices=32, view=None):
        """
        Initialize the dataset.
        
        Args:
            root_dir (str): Project root directory
            task (str): Classification task ('abnormal', 'acl', or 'meniscus')
            split (str): Dataset split ('train', 'valid', or 'test')
            transform (callable, optional): Transform to apply to the data
            max_slices (int): Maximum number of slices to use per MRI volume
            view (str, optional): Specific view to load ('axial', 'coronal', 'sagittal').
                                  If None, all views will be loaded, but this is less efficient.
        """
        self.root_dir = root_dir
        self.task = task
        self.split = split
        self.transform = transform
        self.max_slices = max_slices
        
        # Determine which views to load
        self.views_to_load = [view] if view else ['axial', 'coronal', 'sagittal']
        
        # Load labels based on split
        self.labels_path = self._get_labels_path()
        self.labels = self._load_labels()
        
        # Set up data directories
        self.view_dirs = self._setup_data_directories()
        
        # Log dataset information
        self._log_dataset_info()
        
    def _get_labels_path(self):
        """Get the path to the labels file based on the dataset split."""
        if self.split == 'train':
            return os.path.join(self.root_dir, 'data', 'MRNet-v1.0', f'train-{self.task}-split_train.csv')
        elif self.split == 'valid':
            return os.path.join(self.root_dir, 'data', 'MRNet-v1.0', f'valid-{self.task}.csv')
        elif self.split == 'test':
            return os.path.join(self.root_dir, 'data', 'MRNet-v1.0', f'train-{self.task}-split_test.csv')
        else:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train', 'valid', or 'test'.")
    
    def _load_labels(self):
        """Load and validate the labels file."""
        if not os.path.exists(self.labels_path):
            raise FileNotFoundError(f"Labels file not found: {self.labels_path}")
        
        return pd.read_csv(self.labels_path, header=None, names=['case_id', 'label'])
    
    def _setup_data_directories(self):
        """Set up the directories for each view based on the split."""
        if self.split == 'train':
            data_dir = os.path.join(self.root_dir, 'data', 'MRNet-v1.0', 'processed_train_data')
            return {
                'axial': os.path.join(data_dir, 'axial_train'),
                'coronal': os.path.join(data_dir, 'coronal_train'),
                'sagittal': os.path.join(data_dir, 'sagittal_train')
            }
        elif self.split == 'valid':
            data_dir = os.path.join(self.root_dir, 'data', 'MRNet-v1.0', 'processed_valid_data')
            return {
                'axial': os.path.join(data_dir, 'axial'),
                'coronal': os.path.join(data_dir, 'coronal'),
                'sagittal': os.path.join(data_dir, 'sagittal')
            }
        elif self.split == 'test':
            data_dir = os.path.join(self.root_dir, 'data', 'MRNet-v1.0', 'processed_train_data')
            return {
                'axial': os.path.join(data_dir, 'axial_test'),
                'coronal': os.path.join(data_dir, 'coronal_test'),
                'sagittal': os.path.join(data_dir, 'sagittal_test')
            }
    
    def _log_dataset_info(self):
        """Log information about the dataset."""
        logger.info(f"Dataset initialized: task={self.task}, split={self.split}")
        logger.info(f"Loading views: {self.views_to_load}")
        logger.info(f"Max slices per MRI: {self.max_slices}")
        
        # Verify directories exist
        for view in self.views_to_load:
            dir_path = self.view_dirs[view]
            if not os.path.exists(dir_path):
                logger.warning(f"View directory not found: {dir_path}")
            else:
                file_count = len([f for f in os.listdir(dir_path) if f.endswith('.npy')])
                logger.info(f"Found {file_count} .npy files in {view} directory")
        
        logger.info(f"Loaded {len(self.labels)} cases")
    
    def __len__(self):
        """Return the number of cases in this dataset."""
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample to fetch
            
        Returns:
            dict: Sample dictionary containing case ID, label, available views, and MRI data
        """
        # Get case info
        case_id = self.labels.iloc[idx, 0]
        case_path = f"{int(case_id):04d}.npy"  # Format to match file naming
        label = self.labels.iloc[idx, 1]
        
        # Initialize sample dictionary
        sample = {
            'case_id': case_id, 
            'label': torch.tensor(label, dtype=torch.float32),
            'available_views': []
        }
        
        # Load data for each requested view
        for view in self.views_to_load:
            file_path = os.path.join(self.view_dirs[view], case_path)
            if os.path.exists(file_path):
                try:
                    # Load the data
                    data = np.load(file_path)
                    
                    # Apply slice limiting if needed
                    if data.shape[0] > self.max_slices:
                        # Take center slices (most diagnostically relevant)
                        start_idx = max(0, (data.shape[0] - self.max_slices) // 2)
                        data = data[start_idx:start_idx + self.max_slices]
                    
                    # Apply transform if provided
                    if self.transform:
                        data = self.transform(data)
                    
                    sample[view] = torch.tensor(data, dtype=torch.float32)
                    sample['available_views'].append(view)
                    
                except Exception as e:
                    logger.error(f"Error loading {view} for case {case_id}: {str(e)}")
            
        # Check if we have any views available
        if not sample['available_views']:
            logger.warning(f"No views available for case {case_id} (idx={idx})")
            
        return sample


def custom_collate(batch):
    """
    Custom collation function for single-view MRI data.
    
    This function handles variable-sized 3D MRI volumes by collating 
    only the samples that have a particular view available.
    
    Args:
        batch (list): List of sample dictionaries from the dataset
        
    Returns:
        dict: Collated batch with stacked tensors for each available view
    """
    # Start with collecting simple values
    collated_batch = {
        'case_id': [item['case_id'] for item in batch],
        'label': torch.stack([item['label'] for item in batch]),
    }
    
    # Find all available views across the batch
    all_views = set()
    for item in batch:
        all_views.update(item['available_views'])
    
    collated_batch['available_views'] = list(all_views)
    
    # Collate each view separately
    for view in all_views:
        # Get all tensors for this view
        view_tensors = []
        for item in batch:
            if view in item['available_views']:
                view_tensors.append(item[view])
        
        if view_tensors:
            try:
                # Stack them along the batch dimension
                collated_batch[view] = torch.stack(view_tensors)
            except Exception as e:
                logger.error(f"Error stacking {view} tensors: {str(e)}")
                # Log individual tensor shapes for debugging
                logger.debug("Individual tensor shapes:")
                for i, tensor in enumerate(view_tensors):
                    logger.debug(f"  Tensor {i}: {tensor.shape}, dtype={tensor.dtype}")
    
    return collated_batch


def get_mrnet_dataloader(root_dir, task, view, split='train', transform=None, 
                        batch_size=4, num_workers=1, shuffle=True, max_slices=32):
    """
    Create a DataLoader for single-view MRNet data.
    
    This is a convenience function that sets up both the dataset and dataloader
    with the proper configuration for the single-view approach.
    
    Args:
        root_dir (str): Project root directory
        task (str): Classification task ('abnormal', 'acl', or 'meniscus')
        view (str): The specific view to load ('axial', 'coronal', or 'sagittal')
        split (str): Dataset split ('train', 'valid', or 'test')
        transform (callable, optional): Transform to apply to the data
        batch_size (int): Batch size
        num_workers (int): Number of worker processes for loading data
        shuffle (bool): Whether to shuffle the data
        max_slices (int): Maximum number of slices to use per MRI volume
        
    Returns:
        torch.utils.data.DataLoader: DataLoader for MRNet data
    """
    from torch.utils.data import DataLoader
    
    dataset = MRNetDataset(
        root_dir=root_dir,
        task=task,
        split=split,
        transform=transform,
        max_slices=max_slices,
        view=view  # Explicitly pass the view for single-view approach
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=custom_collate,
        pin_memory=True
    )

def create_data_split(root_dir, task, test_size=0.2, random_state=42):
    """
    Create train/test split from the training data.
    
    Args:
        root_dir (str): Root directory containing the data
        task (str): 'acl', 'meniscus', or 'abnormal'
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
    
    Returns:
        dict: Contains train and test case IDs and labels
    """
    # Construct the labels path using your existing structure
    labels_path = os.path.join(root_dir, 'data', 'MRNet-v1.0', f'train-{task}.csv')
    
    # Verify file exists
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found at {labels_path}")
    
    # Load and verify CSV structure
    try:
        df = pd.read_csv(labels_path)
        # Check if headers exist by looking at column names
        if df.columns.tolist() == ['case_id', 'label']:
            # CSV already has correct headers
            pass
        else:
            # Assume no headers, assign them
            df = pd.read_csv(labels_path, names=['case_id', 'label'])
        
        # Verify data types
        df['label'] = df['label'].astype(float)
        df['case_id'] = df['case_id'].astype(str)
        
    except Exception as e:
        raise ValueError(f"Error reading labels file: {e}")
    
    # Verify we have enough data for splitting
    if len(df) < 10:  # arbitrary minimum size
        raise ValueError(f"Dataset too small to split: {len(df)} samples")
    
    # Verify we have both classes for stratification
    if len(df['label'].unique()) < 2:
        print("Warning: Only one class found in dataset, stratification disabled")
        stratify = None
    else:
        stratify = df['label'].values
    
    # Create the split
    train_cases, test_cases, train_labels, test_labels = train_test_split(
        df['case_id'].values,
        df['label'].values,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )
    
    # Verify split sizes
    print(f"Split sizes - Train: {len(train_cases)}, Test: {len(test_cases)}")
    
    return {
        'train': {'case_ids': train_cases, 'labels': train_labels},
        'test': {'case_ids': test_cases, 'labels': test_labels}
    }