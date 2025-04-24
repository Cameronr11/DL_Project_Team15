#!/usr/bin/env python
"""
Train MRNet on the Stanford MRNet-v1.0 dataset (single-GPU).

Key features
------------
• Attention-weighted mean–max slice pooling (see experiment_model.MRNetModel)
• Backbone frozen for warm-up, then partially unfrozen
• Class-balanced BCEWithLogitsLoss
• AdamW with differential LR (backbone vs. head)
• Mixed-precision + GradScaler
• Three-phase augmentation scheduler
"""

import os, sys, time, gc, argparse, json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score

# ───────────────────────────────────────────────────────────────────────────────
#  Local imports
# ───────────────────────────────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.data_loader import MRNetDataset, custom_collate
from src.data_augmentation_scheduler import DataAugmentationScheduler
from src.data_augmentation import SimpleMRIAugmentation
from src.experiment_model2.MRNetModel import MRNetModel
# ───────────────────────────────────────────────────────────────────────────────


# ╭────────────────────────────────────────────────────────────────────────────╮
# │ Early stopping                                                            │
# ╰────────────────────────────────────────────────────────────────────────────╯
class DualEarlyStopping:
    def __init__(self, patience=7, delta=0.0, verbose=True):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_auc = float('-inf')
        self.best_loss = float('inf')
        self.counter = 0
        self.stop = False

    def __call__(self, val_auc, val_loss, model=None, path=None):
        auc_improved = val_auc > self.best_auc + self.delta
        loss_improved = val_loss < self.best_loss - self.delta

        # Update best values regardless of counter logic
        if auc_improved:
            self.best_auc = val_auc
        if loss_improved:
            self.best_loss = val_loss

        # Only save model when either metric improves
        if auc_improved or loss_improved:
            if model and path:
                torch.save(model.state_dict(), path)
                if self.verbose:
                    print(f"✨  Saved new best model (AUC: {val_auc:.4f}, Loss: {val_loss:.4f})")
        
        # Only increment counter when BOTH metrics fail to improve
        if not auc_improved and not loss_improved:
            self.counter += 1
            if self.verbose:
                print(f"⏳  Early-stop counter {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.stop = True
                print("🛑  Early stopping triggered.")
        else:
            # Reset counter if either metric improves
            self.counter = 0

        return self.stop


# ╭────────────────────────────────────────────────────────────────────────────╮
# │ Utility functions                                                         │
# ╰────────────────────────────────────────────────────────────────────────────╯
def get_project_root():
    return project_root


def class_pos_weight(dataset: MRNetDataset) -> torch.Tensor:
    pos = dataset.labels["label"].mean()
    weight = (1 - pos) / (pos + 1e-6)
    return torch.tensor([weight], dtype=torch.float32)


# ╭────────────────────────────────────────────────────────────────────────────╮
# │ Training & validation loops                                               │
# ╰────────────────────────────────────────────────────────────────────────────╯
def run_epoch(model, loader, criterion, optimizer, scaler,
              device, use_amp, train=True, epoch=0, phase="train",
              log_interval=50):
    if train:
        model.train()
    else:
        model.eval()

    running_loss, y_true, y_pred = 0.0, [], []
    torch.cuda.empty_cache()

    for step, batch in enumerate(loader, 1):
        if model.backbone_type != "ensemble" and model.view not in batch["available_views"]:
            continue

        data   = batch[model.view].to(device, non_blocking=True)
        labels = batch["label"].to(device).view(-1, 1)

        with torch.set_grad_enabled(train), autocast(device_type="cuda", enabled=use_amp):
            outputs = model(data)
            loss    = criterion(outputs, labels)

        if train:
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        running_loss += loss.item()
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(torch.sigmoid(outputs).detach().cpu().numpy())

        # ── per-batch logging ───────────────────────────────────────────────
        if step % log_interval == 0 or step == len(loader):
            avg_loss = running_loss / step
            batch_auc = 0.0
            if len(np.unique(y_true)) > 1:
                try:
                    batch_auc = roc_auc_score(y_true, y_pred)
                except ValueError:
                    batch_auc = 0.0
            print(f"   [{phase}  E{epoch:02d}  B{step:04d}/{len(loader)}] "
                  f"loss {avg_loss:.4f}  AUC {batch_auc:.3f}")
    # ──────────────────────────────────────────────────────────────────────────
    epoch_loss = running_loss / len(loader)
    epoch_auc  = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0
    return epoch_loss, epoch_auc


# ╭────────────────────────────────────────────────────────────────────────────╮
# │ Main training routine                                                     │
# ╰────────────────────────────────────────────────────────────────────────────╯
def train(args) -> float:
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    # ── Data ────────────────────────────────────────────────────────────────
    root = args.data_dir or get_project_root()
    train_ds = MRNetDataset(root, args.task, "train", transform=None,
                            max_slices=args.max_slices, view=args.view)
    val_ds   = MRNetDataset(root, args.task, "valid", transform=None,
                            max_slices=args.max_slices, view=args.view)
    

    #this is an attempt to fix the class imbalance issue however creating a toggle beacause this could be a culprit for the drop in performance accuracy found in 4/22 updates
    if args.no_pos_weight:
        pos_weight = None
    else:
        pos_weight = class_pos_weight(train_ds).to(device)

    criterion = nn.BCEWithLogitsLoss(**({"pos_weight": pos_weight} if pos_weight is not None else {}))

    # Set up data augmentation
    if args.use_da_scheduler:
        print("Using data augmentation scheduler with three phases")
        aug_sched = DataAugmentationScheduler(args.epochs, phase1=0.1, phase2=0.5,
                                           max_rot=12.0, max_brightness=0.12)
        train_ds.set_transform(aug_sched.get_transform(0))
    elif args.use_simple_da:
        print("Using simple data augmentation with fixed parameters")
        simple_aug = SimpleMRIAugmentation(p=0.5, rotation_degrees=8.0, brightness_factor=0.1)
        train_ds.set_transform(simple_aug)
    else:
        print("No data augmentation will be used")
        aug_sched = None
        train_ds.set_transform(None)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              collate_fn=custom_collate)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True,
                              collate_fn=custom_collate)

    # ── Model ───────────────────────────────────────────────────────────────
    model = MRNetModel(backbone=args.backbone, train_backbone=True).to(device)
    model.view = args.view
    unfreeze_at = max(2, args.epochs // 8)  # Unfreeze much earlier

    # Define a second unfreeze point for complete unfreezing
    full_unfreeze_at = max(5, args.epochs // 3)

    head_params     = list(model.classifier.parameters()) + list(model.slice_attn.parameters())
    backbone_params = list(model.feature_extractor.parameters())        # all backbone layers

    optimizer = torch.optim.AdamW(
        [ {"params": backbone_params, "lr": args.lr * 0.01},  # Small but non-zero LR instead of freezing
          {"params": head_params,     "lr": args.lr} ],
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=4, factor=0.5, verbose=True
    )
    scaler  = GradScaler()
    use_amp = True

    # ── Logging ─────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.output_dir, "logs"))
    json.dump(vars(args), open(Path(args.output_dir) / "args.json", "w"), indent=4)

    early_stop = DualEarlyStopping(patience=args.early_stopping_patience,
                                   delta=args.early_stopping_delta,
                                   verbose=True)

    best_val_auc = 0.0
    for epoch in range(args.epochs):
        if args.use_da_scheduler and aug_sched:
            train_ds.set_transform(aug_sched.get_transform(epoch))

        if epoch == unfreeze_at:
            print(f"🔓  Unfreezing top 70% of backbone at epoch {epoch}")
            model.unfreeze(0.70)                          # enable grads on top 70%
            # increase backbone learning rate
            backbone_lr = args.lr * 0.1                   # higher LR for adaptation
            optimizer.param_groups[0]["lr"] = backbone_lr
            print(f"    Backbone learning-rate set to {backbone_lr:g}")

        if epoch == full_unfreeze_at:
            print(f"🔓  Unfreezing all backbone layers at epoch {epoch}")
            model.unfreeze(1.0)                           # enable grads on all layers
            # increase backbone learning rate further
            backbone_lr = args.lr * 0.1                   # higher LR for full adaptation
            optimizer.param_groups[0]["lr"] = backbone_lr
            print(f"    Backbone learning-rate set to {backbone_lr:g}")

        t0 = time.time()
        train_loss, train_auc = run_epoch(model, train_loader, criterion, optimizer, scaler,
                                          device, use_amp, train=True,  epoch=epoch,
                                          phase="train", log_interval=args.log_interval)
        val_loss,   val_auc   = run_epoch(model, val_loader,   criterion, optimizer, scaler,
                                          device, use_amp, train=False, epoch=epoch,
                                          phase="val",   log_interval=args.log_interval)

        scheduler.step(val_loss)

        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("AUC",  {"train": train_auc,  "val": val_auc},  epoch)
        print(f"[{epoch+1:03d}/{args.epochs}]  "
              f"Train loss {train_loss:.4f}  AUC {train_auc:.3f} | "
              f"Val loss {val_loss:.4f}  AUC {val_auc:.3f} | {time.time()-t0:.1f}s")

        if early_stop(val_auc, val_loss, model, Path(args.output_dir) / "best_model.pth"):
            print("⏹️  Early stopping triggered.")
            break

        best_val_auc = max(best_val_auc, val_auc)

    writer.close()
    return best_val_auc


# ╭────────────────────────────────────────────────────────────────────────────╮
# │ Arg-parser                                                                 │
# ╰────────────────────────────────────────────────────────────────────────────╯
def get_parser():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--task", type=str, default="abnormal",
                   choices=["abnormal", "acl", "meniscus"])
    p.add_argument("--view", type=str, required=True,
                   choices=["axial", "coronal", "sagittal"])
    p.add_argument("--max_slices", type=int, default=32)
    # Model
    p.add_argument("--backbone", type=str, default="resnet18",
                   choices=["alexnet", "resnet18", "resnet34", "densenet121"])
    # Training
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--log_interval", type=int, default=50,
                   help="Print batch stats every N batches")
    # Augmentation
    p.add_argument("--use_da_scheduler", action="store_true",
                  help="Use the three-phase data augmentation scheduler")
    p.add_argument("--use_simple_da", action="store_true",
                  help="Use simple data augmentation with fixed parameters")
    # Early stop
    p.add_argument("--early_stopping_patience", type=int, default=3)
    p.add_argument("--early_stopping_delta", type=float, default=0.005)
    # Misc
    #this flag is used to disable class-imbalance weighting in BCE loss
    p.add_argument("--no_pos_weight", action="store_true",
                   help="Disable class-imbalance weighting in BCE loss")
    p.add_argument("--output_dir", type=str, default="results/run1")
    return p


# ╭────────────────────────────────────────────────────────────────────────────╮
# │ Entrypoint                                                                 │
# ╰────────────────────────────────────────────────────────────────────────────╯
if __name__ == "__main__":
    args = get_parser().parse_args()

    # Check for incompatible augmentation options
    if args.use_da_scheduler and args.use_simple_da:
        raise ValueError("Cannot use both --use_da_scheduler and --use_simple_da together. Please choose one.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required.")

    print("🚀  Starting MRNet training")
    best_auc = train(args)
    print(f"✅  Finished. Best validation AUC: {best_auc:.4f}")
