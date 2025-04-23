"""
MRNet Dataset Loader

This module handles loading and preprocessing of MRI knee scan data for the MRNet project.
It provides a PyTorch Dataset for working with single‑view MRI data.

Key components
--------------
- MRNetDataset: PyTorch Dataset for loading MRI data from a specific view (axial, coronal, or sagittal)
- custom_collate:   Collate function that stacks only the views present in each batch
- get_mrnet_dataloader: Convenience wrapper around Dataset + DataLoader
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# -----------------------------------------------------------------------------
#  Logging setup
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
#  Dataset
# ----------------------------------------------------------------------------
class MRNetDataset(Dataset):
    """Single‑view MRNet dataset (axial, coronal or sagittal).

    Only **one** view is loaded per Dataset instance; that keeps memory usage
    low and makes training/validation logic trivial.
    """

    def __init__(
        self,
        root_dir: str | Path,
        task: str,
        split: str = "train",
        transform=None,
        max_slices: int = 32,
        view: str | None = None,
    ):
        if split not in {"train", "valid", "test"}:
            raise ValueError("split must be 'train', 'valid' or 'test'")
        if view not in {None, "axial", "coronal", "sagittal"}:
            raise ValueError("view must be 'axial', 'coronal', 'sagittal' or None")

        self.root_dir   = Path(root_dir)
        self.task       = task
        self.split      = split
        self.transform  = transform
        self.max_slices = max_slices

        # which views to load for *this* Dataset instance
        self.views_to_load: List[str] = [view] if view else ["axial", "coronal", "sagittal"]

        # ------------------------------------------------------------------
        #  Labels & directories
        # ------------------------------------------------------------------
        self.labels_path = self._get_labels_path()
        self.labels      = self._load_labels()
        self.view_dirs: Dict[str, Path] = self._setup_data_directories()

        self._log_dataset_info()

    # ---------------------------------------------------------------------
    #  Helper utilities
    # ---------------------------------------------------------------------
    def _get_labels_path(self) -> Path:
        base = self.root_dir / "data" / "MRNet-v1.0"
        if self.split == "train":
            return base / f"train-{self.task}-split_train.csv"
        if self.split == "valid":
            return base / f"valid-{self.task}.csv"
        return base / f"train-{self.task}-split_test.csv"

    def _load_labels(self) -> pd.DataFrame:
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {self.labels_path}")
        return pd.read_csv(self.labels_path, header=None, names=["case_id", "label"])

    def _setup_data_directories(self) -> Dict[str, Path]:
        base = self.root_dir / "data" / "MRNet-v1.0"
        if self.split == "train":
            d = base / "processed_train_data"
            suffix = {"axial": "axial_train", "coronal": "coronal_train", "sagittal": "sagittal_train"}
        elif self.split == "valid":
            d = base / "processed_valid_data"
            suffix = {"axial": "axial", "coronal": "coronal", "sagittal": "sagittal"}
        else:  # test
            d = base / "processed_train_data"
            suffix = {"axial": "axial_test", "coronal": "coronal_test", "sagittal": "sagittal_test"}
        return {v: d / s for v, s in suffix.items()}

    def _log_dataset_info(self):
        logger.info(f"Dataset initialised: task={self.task}, split={self.split}")
        logger.info(f"Views to load: {self.views_to_load}")
        for view in self.views_to_load:
            n_files = len(list(self.view_dirs[view].glob("*.npy")))
            logger.info(f"  {view:8s}: {n_files} .npy files in {self.view_dirs[view]}")
        logger.info(f"Total cases in CSV: {len(self.labels)}")

    def set_transform(self, transform):
        """
        Allow the training script to swap in a new augmentation / preprocessing
        pipeline (called once per epoch by the scheduler).
        """
        self.transform = transform
        logger.info(f"Updated transform for {self.split} dataset")

    # ---------------------------------------------------------------------
    #  PyTorch Dataset interface
    # ---------------------------------------------------------------------
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        case_id  = int(self.labels.iloc[idx, 0])
        label    = float(self.labels.iloc[idx, 1])
        filename = f"{case_id:04d}.npy"

        sample: Dict[str, torch.Tensor | int | List[str]] = {
            "case_id": case_id,
            "label":   torch.tensor(label, dtype=torch.float32),
            "available_views": [],
        }

        for view in self.views_to_load:
            file_path = self.view_dirs[view] / filename
            if not file_path.exists():
                continue
            try:
                data = np.load(file_path)
                # slice‑limit (take centre slices)
                if data.shape[0] > self.max_slices:
                    start = (data.shape[0] - self.max_slices) // 2
                    data  = data[start : start + self.max_slices]

                # optional transform (augmentation / normalisation)
                if self.transform:
                    data = self.transform(data)

                # convert to float tensor (reuse if already tensor)
                tensor = data.float() if isinstance(data, torch.Tensor) else torch.from_numpy(data).float()

                sample[view] = tensor
                sample["available_views"].append(view)
            except Exception as exc:
                logger.error(f"Error loading view {view} for case {case_id}: {exc}")

        if not sample["available_views"]:
            logger.warning(f"No views available for case {case_id} (idx={idx})")
        return sample

# ----------------------------------------------------------------------------
#  Collate
# ----------------------------------------------------------------------------
def custom_collate(batch: List[Dict]):
    collated = {
        "case_id": [item["case_id"] for item in batch],
        "label":   torch.stack([item["label"] for item in batch]),
    }

    all_views: set[str] = set()
    for item in batch:
        all_views.update(item["available_views"])
    collated["available_views"] = list(all_views)

    for view in all_views:
        tensors = [item[view] for item in batch if view in item["available_views"]]
        if tensors:
            try:
                collated[view] = torch.stack(tensors)
            except Exception as exc:
                logger.error(f"Error stacking view {view}: {exc}")
    return collated

# ----------------------------------------------------------------------------
#  Convenience DataLoader wrapper
# ----------------------------------------------------------------------------

def get_mrnet_dataloader(
    root_dir: str | Path,
    task: str,
    view: str,
    split: str = "train",
    transform=None,
    batch_size: int = 4,
    num_workers: int = 1,
    shuffle: bool = True,
    max_slices: int = 32,
):
    dataset = MRNetDataset(root_dir, task, split, transform, max_slices, view)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate,
    )

# ----------------------------------------------------------------------------
#  (Optional) helper to create a train / test split straight from original CSV
# ----------------------------------------------------------------------------

def create_data_split(root_dir: str | Path, task: str, test_size: float = 0.2, random_state: int = 42):
    """Return dict with 'train' and 'test' keys listing case_ids and labels."""
    csv_path = Path(root_dir) / "data" / "MRNet-v1.0" / f"train-{task}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path, names=["case_id", "label"]) if df_has_no_header(csv_path) else pd.read_csv(csv_path)
    stratify = df["label"] if df["label"].nunique() > 1 else None
    train_c, test_c, train_l, test_l = train_test_split(
        df["case_id"].values,
        df["label"].values,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return {"train": {"case_ids": train_c, "labels": train_l}, "test": {"case_ids": test_c, "labels": test_l}}


def df_has_no_header(path: Path) -> bool:
    """Heuristic: csv with two columns but *no* header row."""
    with path.open() as f:
        first_line = f.readline().strip().split(",")
    return first_line == ["0", "0"] or not any(x.isalpha() for x in first_line)