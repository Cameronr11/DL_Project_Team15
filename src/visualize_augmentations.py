import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.data_normalization import process_series
from src.data_augmentation import SimpleMRIAugmentation
from src.data_augmentation_scheduler import DataAugmentationScheduler


os.makedirs("presentation_visualizations", exist_ok=True)
raw_path = "data/MRNet-v1.0/train/axial/0029.npy"
processed_path = "data/MRNet-v1.0/processed_train_data/axial/0029.npy"
slice_idx = 10
max_slices = 32
image_shape = (224, 224)

# ========== Step 1: Load Raw (Unprocessed) ==========
raw_volume = np.load(raw_path)
raw_volume = np.transpose(raw_volume, (2, 0, 1))  # (S, H, W)
raw_volume = raw_volume[:max_slices]
raw_volume = torch.from_numpy(raw_volume).unsqueeze(1).float()  # (S, 1, H, W)

# ========== Step 2: Load Processed ==========
processed_volume = np.load(processed_path)  # already (S, C, H, W)
processed_volume = torch.tensor(processed_volume).float()


# ========== Step 3: Simple Augmentation ==========
simple_aug = SimpleMRIAugmentation(p=1.0, rotation_degrees=10, brightness_factor=0.1)
simple_volume = simple_aug(processed_volume.clone())

# ========== Step 4: Scheduler (Epoch 5 and 20) ==========
scheduler = DataAugmentationScheduler(total_epochs=25, max_rot=15.0, max_brightness=0.12)

aug_phase2 = scheduler.get_transform(epoch=5)
aug_phase3 = scheduler.get_transform(epoch=20)

sched_phase2_volume = aug_phase2(processed_volume.clone())
sched_phase3_volume = aug_phase3(processed_volume.clone())

# ========== Step 5: Visualization ==========
def plot_comparison(volumes, titles, slice_idx=10, output_path="presentation_visualizations/augmentation_comparison.png"):
    fig, axs = plt.subplots(1, len(volumes), figsize=(20, 5))
    for i, vol in enumerate(volumes):
        slice_img = vol[slice_idx]
        if slice_img.shape[0] == 1:
            img = slice_img.squeeze(0).numpy()  # Grayscale from raw
        else:
            img = slice_img.permute(1, 2, 0).numpy()  # RGB from processed
        axs[i].imshow(img, cmap='gray')
        axs[i].set_title(titles[i])
        axs[i].axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

plot_comparison(
    volumes=[
        raw_volume, 
        processed_volume, 
        simple_volume, 
        sched_phase2_volume, 
        sched_phase3_volume
    ],
    titles=[
        "Raw Slice (Unprocessed)",
        "Processed Slice",
        "Simple Augmentation",
        "Scheduler (Epoch 5)",
        "Scheduler (Epoch 20)"
    ],
    slice_idx=slice_idx
)