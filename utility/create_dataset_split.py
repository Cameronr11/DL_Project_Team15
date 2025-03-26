import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
import shutil

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def get_project_root():
    """Returns the absolute path to the project root directory"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def create_and_organize_split(root_dir, test_size=0.2, random_state=42):
    """
    Create train/test splits for both data files and labels,
    keeping validation data separate
    """
    data_dir = os.path.join(root_dir, 'data', 'MRNet-v1.0')
    processed_train_dir = os.path.join(data_dir, 'processed_train_data')
    
    print(f"Processing data in: {data_dir}")
    print(f"Train data location: {processed_train_dir}")
    
    # First, get all case IDs from any task (they should all have the same cases)
    df_abnormal = pd.read_csv(os.path.join(data_dir, 'train-abnormal.csv'), names=['case_id', 'label'])
    all_case_ids = df_abnormal['case_id'].values
    
    # Create the case ID split first
    train_cases, test_cases = train_test_split(
        all_case_ids,
        test_size=test_size,
        random_state=random_state
    )
    
    # Now process each task using the same case ID split
    tasks = ['abnormal', 'acl', 'meniscus']
    for task in tasks:
        print(f"\nProcessing labels for task: {task}")
        # Load training labels
        train_labels_path = os.path.join(data_dir, f'train-{task}.csv')
        df = pd.read_csv(train_labels_path, names=['case_id', 'label'])
        
        # Split based on the pre-determined case IDs
        train_df = df[df['case_id'].isin(train_cases)]
        test_df = df[df['case_id'].isin(test_cases)]
        
        # Save split labels
        train_df.to_csv(os.path.join(data_dir, f'train-{task}-split_train.csv'), index=False, header=False)
        test_df.to_csv(os.path.join(data_dir, f'train-{task}-split_test.csv'), index=False, header=False)
        
        print(f"Created label splits for {task}:")
        print(f"Train samples: {len(train_df)}, Positive cases: {train_df['label'].sum()}")
        print(f"Test samples: {len(test_df)}, Positive cases: {test_df['label'].sum()}")
    
    # Convert case IDs to the format used in filenames
    train_cases = [f"{int(case_id):04d}" for case_id in train_cases]
    test_cases = [f"{int(case_id):04d}" for case_id in test_cases]
    
    # Create new directory structure and move files
    views = ['axial', 'coronal', 'sagittal']
    for view in views:
        print(f"\nProcessing {view} view data...")
        
        # Create train directory
        train_dir = os.path.join(processed_train_dir, f"{view}_train")
        os.makedirs(train_dir, exist_ok=True)
        
        # Create test directory
        test_dir = os.path.join(processed_train_dir, f"{view}_test")
        os.makedirs(test_dir, exist_ok=True)
        
        # Source directory
        source_dir = os.path.join(processed_train_dir, view)
        
        if not os.path.exists(source_dir):
            raise FileNotFoundError(f"Source directory not found: {source_dir}")
        
        # Copy train files
        for case_id in train_cases:
            source_file = os.path.join(source_dir, f"{case_id}.npy")
            dest_file = os.path.join(train_dir, f"{case_id}.npy")
            if os.path.exists(source_file):
                shutil.copy2(source_file, dest_file)
            else:
                print(f"Warning: Missing file {source_file}")
        
        # Copy test files
        for case_id in test_cases:
            source_file = os.path.join(source_dir, f"{case_id}.npy")
            dest_file = os.path.join(test_dir, f"{case_id}.npy")
            if os.path.exists(source_file):
                shutil.copy2(source_file, dest_file)
            else:
                print(f"Warning: Missing file {source_file}")
        
        print(f"{view} data split complete:")
        print(f"Train samples: {len(os.listdir(train_dir))}")
        print(f"Test samples: {len(os.listdir(test_dir))}")

if __name__ == "__main__":
    project_root = get_project_root()
    create_and_organize_split(project_root)
