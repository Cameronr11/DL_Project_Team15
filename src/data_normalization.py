import os
import numpy as np
import cv2
import torch

def process_series(npy_path, target_shape=(224, 224), approach='2D', channels=3):
    """
    Load and preprocess an MRI series from a .npy file.
    
    For MRNet, each .npy file is a 3D volume. Typically, the file is shaped (H, W, num_slices).
    This function transposes it to (num_slices, H, W) and then processes each slice.
    
    Args:
        npy_path (str): Path to the .npy file.
        target_shape (tuple): Desired (height, width) of each slice.
        approach (str): '2D' to prepare for slice-based models.
        channels (int): Number of channels expected by the model.
    
    Returns:
        processed_volume (np.array):
            For a 2D approach: shape (num_slices, 3, H, W)
    """
    volume = np.load(npy_path)  # Load the volume
    
    # Check the shape to determine if the last dimension is number of slices.
    # If shape is (H, W, S) and S > 3, assume it's a series of slices.
    # I DONT KNOW IF THIS IF STATEMENT IS NECESSARY BECUASE DATA IS ALREADY IN THE CORRECT FORMAT
    if volume.ndim == 3 and volume.shape[-1] > 3:
        volume = np.transpose(volume, (2, 0, 1))  # Now shape is (num_slices, H, W)
    
    processed_slices = []
    for slice_img in volume:
        # Normalize the slice to [0, 1]
        slice_norm = (slice_img - np.min(slice_img)) / (np.max(slice_img) - np.min(slice_img) + 1e-8)
        slice_norm = np.clip(slice_norm, 0, 1)
        
        # Resize the slice to the target resolution
        slice_resized = cv2.resize(slice_norm, target_shape)
        
        # Process for 2D approach: if model expects 3 channels, replicate the slice
        if approach == '2D':
            if channels == 3:
                slice_processed = np.stack([slice_resized] * 3, axis=-1)  # (H, W, 3)
            else:
                slice_processed = np.expand_dims(slice_resized, axis=-1)  # (H, W, 1)
        else:
            slice_processed = slice_resized  # For other approaches, customize as needed
        
        # Convert to float32
        slice_processed = slice_processed.astype(np.float32)
        # Rearrange to (channels, H, W) if needed (we'll do that after stacking slices)
        processed_slices.append(slice_processed)
    
    # Stack slices into a volume: (num_slices, H, W, channels)
    processed_volume = np.stack(processed_slices, axis=0)
    
    # For compatibility with PyTorch 2D models, convert each slice to (channels, H, W)
    if approach == '2D':
        processed_volume = np.transpose(processed_volume, (0, 3, 1, 2))
    
    return processed_volume

def test_preprocessed_format(file_path, expected_slice_shape=(3, 224, 224)):
    """
    Test a preprocessed file by loading it, verifying its dimensions,
    checking pixel normalization, and converting one slice to a PyTorch tensor.
    
    Args:
        file_path (str): Path to the preprocessed .npy file.
        expected_slice_shape (tuple): Expected shape for each processed slice, e.g. (3, 224, 224)
    """
    # Load a processed volume; expected shape is (num_slices, channels, H, W)
    volume = np.load(file_path)
    num_slices = volume.shape[0]
    
    print(f"Processed volume has {num_slices} slices.")
    sample_slice = volume[0]  # Grab one slice to inspect
    
    # Check slice shape
    if sample_slice.shape != expected_slice_shape:
        print(f"Warning: A slice shape {sample_slice.shape} does not match expected {expected_slice_shape}.")
    else:
        print(f"Slice shape is correct: {sample_slice.shape}.")
    
    # Check pixel normalization in the sample slice
    if np.min(sample_slice) < 0 or np.max(sample_slice) > 1:
        print(f"Warning: Pixel values are out of the [0, 1] range (min: {np.min(sample_slice)}, max: {np.max(sample_slice)}).")
    else:
        print("Pixel values are within the expected range [0, 1].")
    
    # Convert one slice to a PyTorch tensor to verify conversion (adding a batch dimension)
    img_tensor = torch.tensor(sample_slice)
    img_tensor = img_tensor.unsqueeze(0)  # Shape: (1, channels, H, W)
    print("Converted tensor shape (with batch dimension):", img_tensor.shape)
    
    return img_tensor

if __name__ == '__main__':
    # Define source and destination directories
    data_dir = os.path.join('..', 'data', 'MRNet-v1.0', 'train')  # Adjust as needed
    processed_dir = 'processed_data'
    
    # Create the main processed_data directory
    os.makedirs(processed_dir, exist_ok=True)
    
    # List of folders to process (e.g., axial, coronal, sagittal)
    folders = ['axial', 'coronal', 'sagittal']
    
    # Process each folder
    for folder in folders:
        source_folder = os.path.join(data_dir, folder)
        processed_folder = os.path.join(processed_dir, folder)
        os.makedirs(processed_folder, exist_ok=True)
        
        # Process each file in the folder
        for filename in os.listdir(source_folder):
            if filename.endswith('.npy'):
                input_path = os.path.join(source_folder, filename)
                output_path = os.path.join(processed_folder, filename)
                
                try:
                    # Print the original file size
                    original_volume = np.load(input_path)
                    print(f"Processing {filename}, original shape: {original_volume.shape}")
                    
                    # Process the MRI series
                    processed_volume = process_series(input_path, target_shape=(224, 224), approach='2D', channels=3)
                    print(f"Processed shape: {processed_volume.shape}, size: {processed_volume.nbytes / (1024*1024):.2f} MB")
                    
                    # Save the processed volume as a .npy file
                    np.save(output_path, processed_volume)
                    print(f"Processed and saved: {output_path}")
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    continue
    
    # Test one of the preprocessed files (adjust the filename as needed)
    test_file_path = os.path.join(processed_dir, 'axial', '0029.npy')
    if os.path.exists(test_file_path):
        test_preprocessed_format(test_file_path, expected_slice_shape=(3, 224, 224))
    else:
        print(f"Sample file {test_file_path} not found.")
