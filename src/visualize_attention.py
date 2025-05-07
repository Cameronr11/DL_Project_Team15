import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.experiment_model.MRNetModel import MRNetModel
from src.data_normalization import process_series

# ========== Config ==========
MODEL_PATH = "src/results/best_model_attempt/resnet18_da_schedule_no_weight_decay/abnormal_axial/best_model.pth"
RAW_DATA_PATH  = "data/MRNet-v1.0/processed_train_data/axial/0029.npy"
MAX_SLICES = 32
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ========== Load Data ==========
volume = np.load(RAW_DATA_PATH)  # Already preprocessed: (S, C, H, W)
volume_tensor = torch.tensor(volume).unsqueeze(0).float().to(DEVICE)  # (1, S, C, H, W)


# ========== Load Model ==========
model = MRNetModel(backbone="resnet18", use_attention=True).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ========== Modify forward to return attention ==========
def forward_with_attention(model, x):
    B, S, C, H, W = x.shape
    x = x.view(B * S, C, H, W)
    feats = model.feature_extractor(x)
    feats = model.global_pool(feats).flatten(1)
    feats = feats.view(B, S, -1)

    attn = model.slice_attn(feats).softmax(dim=1)
    mean_pool = (feats * attn).sum(dim=1)
    max_pool = feats.max(dim=1).values
    pooled = torch.cat([mean_pool, max_pool], dim=1)
    out = model.classifier(pooled)
    return out, attn

# ========== Run Inference & Get Attention ==========
with torch.no_grad():
    _, attn_weights = forward_with_attention(model, volume_tensor)

# ========== Plot Attention ==========
def plot_attention_weights(attn_weights, save_path="slice_attention.png"):
    if attn_weights.ndim == 3:
        attn_weights = attn_weights.squeeze(0).squeeze(-1)  # (S,)

    plt.figure(figsize=(10, 4))
    plt.plot(attn_weights.cpu().numpy(), marker='o')
    plt.title("Slice Attention Weights")
    plt.xlabel("Slice Index")
    plt.ylabel("Attention Weight")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

plot_attention_weights(attn_weights)
