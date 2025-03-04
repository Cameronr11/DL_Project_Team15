import torch
import pandas as pd
import numpy as np
import os
import glob
from torch.utils.data import Dataset, DataLoader

class MRNetDataset(Dataset):
    """
    Dataset for MRNet knee MRI data.
    
    This dataset handles cases where not all views (axial, coronal, sagittal) 
    may be available for every case ID. The dataset returns only the views
    that are available for each case.
    """
    def __init__(self, root_dir, task, train=True, transform=None, require_all_views=False):
        """
        Args:
            root_dir (str): Root directory of the project
            task (str): 'acl', 'meniscus', or 'abnormal'
            train (bool): If True, use training data, else validation
            transform: Optional transforms to apply to images
            require_all_views (bool): If True, only include cases with all three views
        """
        self.root_dir = root_dir
        self.task = task
        self.train = train
        self.transform = transform
        
        # Set up paths
        split = 'train' if train else 'valid'
        
        # Set up the data directory path
        self.data_dir = os.path.join(root_dir, 'data', 'MRNet-v1.0', f'processed_{split}_data')
        
        # Verify the directory exists
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(
                f"Processed data directory not found at {self.data_dir}. "
                "Please ensure you have run the data normalization script first."
            )
        print(f"Using processed data from: {self.data_dir}")
        
        # Load labels
        labels_file = f"{split}-{task}.csv"
        labels_path = os.path.join(root_dir, 'data', 'MRNet-v1.0', labels_file)
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found at {labels_path}")
        
        self.labels_df = pd.read_csv(labels_path, header=None)
        
        # Get case IDs
        case_ids = self.labels_df.iloc[:, 0].values
        labels = self.labels_df.iloc[:, 1].values
            
        # Discover available case files for each view
        self.available_files = self._discover_available_files()
        
        if require_all_views:
            valid_indices = []
            for i, case_id in enumerate(case_ids):
                if self._has_all_views(case_id):
                    valid_indices.append(i)
            
            if valid_indices:
                self.case_ids = case_ids[valid_indices]
                self.labels = labels[valid_indices]
                print(f"Filtered from {len(case_ids)} to {len(self.case_ids)} cases with all views")
            else:
                print("WARNING: No cases found with all views available. Using all cases.")
                self.case_ids = case_ids
                self.labels = labels
        else:
            self.case_ids = case_ids
            self.labels = labels
    
    def _discover_available_files(self):
        """Discover what files are available for each view"""
        available_files = {'axial': {}, 'coronal': {}, 'sagittal': {}}
        
        for view in ['axial', 'coronal', 'sagittal']:
            view_dir = os.path.join(self.data_dir, view)
            if os.path.exists(view_dir):
                # Get all .npy files in this view directory
                file_pattern = os.path.join(view_dir, "*.npy")
                files = glob.glob(file_pattern)
                
                for file_path in files:
                    # Extract case ID from filename
                    filename = os.path.basename(file_path)
                    case_id = int(os.path.splitext(filename)[0])  # Remove .npy extension and convert to int
                    available_files[view][case_id] = file_path
                
                print(f"Found {len(files)} {view} files in {view_dir}")
            else:
                print(f"WARNING: {view} directory not found at {view_dir}")
        
        return available_files
    
    def _has_all_views(self, case_id):
        """Check if all views are available for this case ID"""
        case_id_int = int(case_id)
        return (case_id_int in self.available_files['axial'] and 
                case_id_int in self.available_files['coronal'] and 
                case_id_int in self.available_files['sagittal'])
        
    def __len__(self):
        return len(self.case_ids)
    
    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        label = self.labels[idx]
        case_id_int = int(case_id)
        
        # Initialize result dictionary with label and case_id
        result = {
            'label': torch.tensor(label, dtype=torch.float32),
            'case_id': case_id,
            'available_views': []
        }
        
        # Try different formats for the case ID
        for view in ['axial', 'coronal', 'sagittal']:
            # First check if it's in our discovered files
            if hasattr(self, 'available_files') and case_id_int in self.available_files[view]:
                file_path = self.available_files[view][case_id_int]
                try:
                    data = np.load(file_path)
                    tensor = torch.from_numpy(data)
                    if self.transform:
                        tensor = self.transform(tensor)
                    result[view] = tensor
                    result['available_views'].append(view)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
            else:
                # If not found in discovered files, try with zero-padding to 4 digits
                formatted_case_id = f"{case_id_int:04d}"
                path = os.path.join(self.data_dir, view, f"{formatted_case_id}.npy")
                
                if os.path.exists(path):
                    try:
                        data = np.load(path)
                        tensor = torch.from_numpy(data)
                        if self.transform:
                            tensor = self.transform(tensor)
                        result[view] = tensor
                        result['available_views'].append(view)
                    except Exception as e:
                        print(f"Error loading {path}: {e}")
        
        # If no views are available, print a warning
        if not result['available_views']:
            # Less verbose warning to avoid spamming
            if idx < 20:  # Only print for first 20 cases
                print(f"Warning: No views available for case {case_id}")
        
        return result
    
    def custom_collate(batch):
        """
        Custom collate function for DataLoader with variable slices.
        Handles cases where different samples might have different views available.
        
        Args:
            batch: List of samples from MRNetDataset
            
        Returns:
            Dictionary containing:
                - label: Stacked labels
                - case_id: List of case IDs
                - available_views: List of views available in the batch
                - view tensors: Stacked tensors for each available view
        """
        # Find common views available in all samples
        available_in_all = set(['axial', 'coronal', 'sagittal'])
        for sample in batch:
            available_in_sample = set(sample['available_views'])
            available_in_all = available_in_all.intersection(available_in_sample)
        
        # Create batch with only common views and necessary data
        result = {
            'label': torch.stack([sample['label'] for sample in batch]),
            'case_id': [sample['case_id'] for sample in batch],
            'available_views': list(available_in_all)
        }
        
        # Add tensors for available views (even if not common to all)
        all_available_views = set()
        for sample in batch:
            all_available_views.update(sample['available_views'])
        
        for view in all_available_views:
            # Get samples that have this view
            valid_samples = [sample[view] for sample in batch if view in sample['available_views']]
            if valid_samples:
                result[view] = torch.stack(valid_samples)
        
        return result