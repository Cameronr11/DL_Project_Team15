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
import matplotlib.pyplot as plt

# ───────────────────────────────────────────────────────────────────────────────
#  Local imports
# ───────────────────────────────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.data_loader import MRNetDataset, custom_collate
from src.data_augmentation_scheduler import DataAugmentationScheduler
from src.data_augmentation import SimpleMRIAugmentation
from src.experiment_model.MRNetModel import MRNetModel, MRNetEnsemble
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
def run_epoch(model, loader, criterion, optimizer, scheduler, scaler,
              device, use_amp, train=True, epoch=0, phase="train",
              log_interval=50):
    """
    One pass over the dataloader.

    Works with:
      • single-view MRNetModel   (expects a 5-D tensor)      -- default path
      • TripleMRNet             (expects a dict of 3 views)
      • any other “ensemble” that accepts a dict of views and has
        model.backbone_type == "ensemble"
    """
    # ------ mode ------------------------------------------------------------
    model.train() if train else model.eval()

    running_loss, y_true, y_pred = 0.0, [], []
    torch.cuda.empty_cache()

    # ------ batches ---------------------------------------------------------
    for step, batch in enumerate(loader, 1):

        # ── 1. Build INPUTS --------------------------------------------------
        if isinstance(model, MRNetEnsemble):
            # need all three views for triple-view training
            needed = ["axial", "coronal", "sagittal"]
            if not all(v in batch["available_views"] for v in needed):
                continue                       # skip incomplete case

            inputs = {v: batch[v].to(device, non_blocking=True)
                      for v in needed}

        elif getattr(model, "backbone_type", None) == "ensemble":
            # generic dict-style ensemble: pass whatever views exist
            inputs = {v: batch[v].to(device, non_blocking=True)
                      for v in batch["available_views"]}

        else:
            # single-view path (original behaviour)
            if model.view not in batch["available_views"]:
                continue
            inputs = batch[model.view].to(device, non_blocking=True)

        # ── 2. Labels --------------------------------------------------------
        labels = batch["label"].to(device).view(-1, 1)

        # ── 3. Forward + loss ------------------------------------------------
        with torch.set_grad_enabled(train), \
             autocast(device_type="cuda", enabled=use_amp):
            outputs = model(inputs)
            loss    = criterion(outputs, labels)

        # ── 4. Optimiser step (if training) ----------------------------------
        if train:
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

        # ── 5. Track metrics -------------------------------------------------
        running_loss += loss.item()
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(torch.sigmoid(outputs).detach().cpu().numpy())

        # ------ per-batch logging -------------------------------------------
        if step % log_interval == 0 or step == len(loader):
            avg_loss  = running_loss / step
            batch_auc = 0.0
            if len(np.unique(y_true)) > 1:
                try:
                    batch_auc = roc_auc_score(y_true, y_pred)
                except ValueError:
                    batch_auc = 0.0
            print(f"   [{phase}  E{epoch:02d}  B{step:04d}/{len(loader)}] "
                  f"loss {avg_loss:.4f}  AUC {batch_auc:.3f}")

    # ------ epoch summary ----------------------------------------------------
    epoch_loss = running_loss / len(loader)
    epoch_auc  = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0
    return epoch_loss, epoch_auc


# ╭────────────────────────────────────────────────────────────────────────────╮
# │ Main training routine                                                     │
# ╰────────────────────────────────────────────────────────────────────────────╯
def train(args) -> float:


    #adding this inner function to plot the training and validation losses and aucs
    def plot_training_curves(train_vals, val_vals, ylabel, save_path):
        plt.figure(figsize=(8, 6))
        epochs = list(range(1, len(train_vals) + 1))
        plt.plot(epochs, train_vals, label='Train', marker='o')
        plt.plot(epochs, val_vals, label='Validation', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.title(f'{ylabel} over Epochs')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    train_losses = []
    val_losses = []
    train_aucs = []
    val_aucs = []
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    # ── Data ────────────────────────────────────────────────────────────────
    root = args.data_dir or get_project_root()
    view_arg = None if args.train_approach == "ensemble" else args.view
    train_ds = MRNetDataset(root, args.task, "train", transform=None,
                            max_slices=args.max_slices, view=view_arg)
    val_ds   = MRNetDataset(root, args.task, "valid", transform=None,
                            max_slices=args.max_slices, view=view_arg)
    

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
    if args.train_approach == "ensemble":
        model = MRNetEnsemble(backbone=args.backbone,
                            train_backbone=False,
                            use_attention=not args.no_attention).to(device)
    else:
        model = MRNetModel(backbone=args.backbone,
                        train_backbone=False,
                        use_attention=not args.no_attention).to(device)
        model.view = args.view or "axial"

    # -----------------------------------------------------------------------
    #  Backbone-training policy  (← new with --backbone_mode)
    # -----------------------------------------------------------------------
    if args.backbone_mode == "full":
        for p in model.parameters():                     # everything trainable
            p.requires_grad = True
        BACKBONE_LR      = args.lr * 0.1                # but still lower than head
        unfreeze_at      = full_unfreeze_at = 10**9     # never trigger later
        print("🟢  Training the entire backbone from epoch 0")

    elif args.backbone_mode == "frozen":
        BACKBONE_LR      = 0.0                          # stays frozen
        unfreeze_at      = full_unfreeze_at = 10**9
        print("🔒  Backbone will remain frozen for all epochs")

    else:                                               # "partial"  (default)
        BACKBONE_LR      = args.lr * 0.01               # tiny LR until unfreeze
        unfreeze_at      = max(2,  args.epochs // 8)    # same as before
        full_unfreeze_at = max(5,  args.epochs // 3)
        print(f"🌓  Partial unfreeze: at {unfreeze_at} / {full_unfreeze_at}")

    # -----------------------------------------------------------------------
    #  Parameter groups
    # -----------------------------------------------------------------------
    # ─── 1. Split backbone vs head parameters ────────────────────────────────
    if args.train_approach == "ensemble":
        head_params = list(model.classifier.parameters())
        for sub in (model.axial, model.coronal, model.sagittal):
            head_params += list(sub.slice_attn.parameters())
        backbone_params = []
        for sub in (model.axial, model.coronal, model.sagittal):
            backbone_params += list(sub.feature_extractor.parameters())
    else:
        head_params     = list(model.classifier.parameters()) + list(model.slice_attn.parameters())
        backbone_params = list(model.feature_extractor.parameters())

    # ─── 2. Learning-rate strategy driven by CLI --lr ────────────────────────
    HEAD_MAX_LR     = args.lr               # e.g. 3e-4
    BACKBONE_MAX_LR = args.lr / 3.0         # e.g. 1e-4

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": BACKBONE_MAX_LR},
            {"params": head_params,     "lr": HEAD_MAX_LR},
        ],
        weight_decay=args.weight_decay,
    )

    # One-Cycle LR schedule: warm-up 10 % then cosine decay to 0
    steps_per_epoch = len(train_loader)
    total_steps     = args.epochs * steps_per_epoch

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[BACKBONE_MAX_LR, HEAD_MAX_LR],  # order matches param groups
        total_steps=total_steps,
        pct_start=0.10,
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
            model.unfreeze(0.70)
            # don’t change learning-rate – One-CycleLR keeps control

        if epoch == full_unfreeze_at:
            print(f"🔓  Unfreezing all backbone layers at epoch {epoch}")
            model.unfreeze(1.0)                           # enable grads on all layers
            # increase backbone learning rate further
            backbone_lr = args.lr * 0.1                   # higher LR for full adaptation
            optimizer.param_groups[0]["lr"] = backbone_lr
            print(f"    Backbone learning-rate set to {backbone_lr:g}")

        t0 = time.time()
        train_loss, train_auc = run_epoch(
            model, train_loader, criterion,
            optimizer, scheduler, scaler,      # ← correct order
            device, use_amp, train=True, epoch=epoch,
            phase="train", log_interval=args.log_interval
        )

        # VAL
        val_loss, val_auc = run_epoch(
            model, val_loader, criterion,
            optimizer, None, scaler,           # scheduler=None for validation
            device, use_amp, train=False, epoch=epoch,
            phase="val", log_interval=args.log_interval
        )

        

        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("AUC",  {"train": train_auc,  "val": val_auc},  epoch)
        print(f"[{epoch+1:03d}/{args.epochs}]  "
              f"Train loss {train_loss:.4f}  AUC {train_auc:.3f} | "
              f"Val loss {val_loss:.4f}  AUC {val_auc:.3f} | {time.time()-t0:.1f}s")
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_aucs.append(train_auc)
        val_aucs.append(val_auc)




        if early_stop(val_auc, val_loss, model, Path(args.output_dir) / "best_model.pth"):
            print("⏹️  Early stopping triggered.")
            break

        best_val_auc = max(best_val_auc, val_auc)



    #plotting the training and validation losses and aucs as visualizations
    plot_training_curves(train_losses, val_losses, "Loss", os.path.join(args.output_dir, "loss_curve.png"))
    plot_training_curves(train_aucs, val_aucs, "AUC", os.path.join(args.output_dir, "auc_curve.png"))



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
    p.add_argument("--view", type=str, default=None,
                   choices=["axial", "coronal", "sagittal"])
    p.add_argument("--max_slices", type=int, default=32)
    p.add_argument("--train_approach", default="per_view", choices=["per_view", "ensemble"])
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
    p.add_argument("--early_stopping_patience", type=int, default=4)
    p.add_argument("--early_stopping_delta", type=float, default=0.003)
    # Misc
    #this flag is used to disable class-imbalance weighting in BCE loss
    p.add_argument("--no_pos_weight", action="store_true",
                   help="Disable class-imbalance weighting in BCE loss")
    p.add_argument("--output_dir", type=str, default="results/run1")
    p.add_argument("--no_attention", action="store_true",
               help="Disable slice attention and use uniform pooling instead")
    
    p.add_argument("--backbone_mode", type=str, default="partial", choices=["partial", "full", "frozen"],)

    return p


# ╭────────────────────────────────────────────────────────────────────────────╮
# │ Entrypoint                                                                 │
# ╰────────────────────────────────────────────────────────────────────────────╯
if __name__ == "__main__":
    args = get_parser().parse_args()
    if args.train_approach == "per_view" and args.view is None:
        raise ValueError("--view must be provided when --train_approach is 'per_view'")
    # Check for incompatible augmentation options
    if args.use_da_scheduler and args.use_simple_da:
        raise ValueError("Cannot use both --use_da_scheduler and --use_simple_da together. Please choose one.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required.")

    print("🚀  Starting MRNet training")
    best_auc = train(args)
    print(f"✅  Finished. Best validation AUC: {best_auc:.4f}")
