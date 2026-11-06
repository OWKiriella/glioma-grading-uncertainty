"""Train the 3D U-Net that segments the whole tumour (BraTS labels 1, 2 and 4
merged into one class). Trained from scratch on the training split.
Checkpoints every epoch and can continue with --resume.

Usage: python train_seg.py --cache-dir CACHE --out-dir RUNS/seg
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

import folds
import monai
import numpy as np
import pandas as pd
import torch
from monai.data import DataLoader, Dataset
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import (CenterSpatialCropd, Compose, RandFlipd, RandGaussianNoised,
                              RandRotate90d, RandScaleIntensityd, RandShiftIntensityd)

# --- Augmentation settings (quoted in Methods 3.5) -------------------------
FLIP_PROB = 0.5           # per axis, all three axes
ROTATE90_PROB = 0.5       # axial plane only
SCALE_INTENSITY_PROB, SCALE_INTENSITY_FACTOR = 0.5, 0.1
SHIFT_INTENSITY_PROB, SHIFT_INTENSITY_OFFSET = 0.5, 0.1
NOISE_PROB, NOISE_STD = 0.15, 0.1

IMAGE_SHAPE = (4, 128, 128, 96)


class CachedBraTS(Dataset):
    """One preprocessed .npz per patient. The mask is binarised to whole tumour."""

    def __init__(self, cache_dir, files, transform=None, crop=None):
        self.cache_dir = cache_dir
        self.files = list(files)
        self.transform = transform
        self.crop = crop          # smoke-test only: centre crop to a smaller volume

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with np.load(os.path.join(self.cache_dir, self.files[index])) as saved:
            image = saved["image"].astype(np.float32)          # [4, 128, 128, 96]
            seg = (saved["seg"] > 0).astype(np.float32)[None]  # [1, 128, 128, 96]

        sample = {"image": torch.from_numpy(image), "seg": torch.from_numpy(seg)}
        if self.crop is not None:
            sample = CenterSpatialCropd(keys=["image", "seg"], roi_size=self.crop)(sample)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def build_augmentation():
    """Training-only transforms. Spatial ones act on the image and mask together."""
    keys = ["image", "seg"]
    return Compose([
        RandFlipd(keys=keys, prob=FLIP_PROB, spatial_axis=0),
        RandFlipd(keys=keys, prob=FLIP_PROB, spatial_axis=1),
        RandFlipd(keys=keys, prob=FLIP_PROB, spatial_axis=2),
        RandRotate90d(keys=keys, prob=ROTATE90_PROB, spatial_axes=(0, 1)),
        RandScaleIntensityd(keys="image", factors=SCALE_INTENSITY_FACTOR, prob=SCALE_INTENSITY_PROB),
        RandShiftIntensityd(keys="image", offsets=SHIFT_INTENSITY_OFFSET, prob=SHIFT_INTENSITY_PROB),
        RandGaussianNoised(keys="image", prob=NOISE_PROB, std=NOISE_STD),
    ])


def build_model(dropout):
    """MONAI 3D U-Net with one output channel (whole tumour)."""
    return monai.networks.nets.UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=1,                      # one foreground class: whole tumour
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=dropout,
    )


def load_split(cache_dir, split, limit=0, fold=None, folds_file=None):
    """File names for one split."""
    rows = folds.resolve_split(cache_dir, split, fold, folds_file, limit)
    print("  " + folds.describe(split, fold, rows))
    return rows["file"].tolist()


def run_validation(model, loader, device, metric, use_amp):
    """Mean Dice over the validation split, at a 0.5 probability threshold."""
    model.eval()
    metric.reset()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["seg"].to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=use_amp):
                logits = model(images)
            predicted = (torch.sigmoid(logits.float()) > 0.5).float()
            metric(y_pred=predicted, y=targets)
    return float(metric.aggregate().item())


def save_checkpoint(path, epoch, model, optimiser, scaler, best_dice, history):
    tmp_path = path + ".tmp"          # never leave a half-written checkpoint
    torch.save({"epoch": epoch,
                "model": model.state_dict(),
                "optimiser": optimiser.state_dict(),
                "scaler": scaler.state_dict(),
                "best_dice": best_dice,
                "history": history}, tmp_path)
    os.replace(tmp_path, path)


def main():
    parser = argparse.ArgumentParser(description="Train the Stage 1 whole-tumour segmentation model.")
    parser.add_argument("--cache-dir", required=True, help="Folder of .npz files from prepare_data.py")
    parser.add_argument("--out-dir", required=True, help="Where to write checkpoints and the log")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Patients per training step (fits a 48 GB L40S)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    parser.add_argument("--seed", type=int, default=42)
    folds.add_fold_arguments(parser, os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable mixed precision (slower, uses more memory)")
    parser.add_argument("--resume", action="store_true",
                        help="Continue from last.pt in --out-dir if it exists")
    parser.add_argument("--limit", type=int, default=0,
                        help="Use only the first N patients per split (0 = all). For debugging.")
    parser.add_argument("--crop", type=int, nargs=3, default=None, metavar=("X", "Y", "Z"),
                        help="Centre-crop every volume to this size. Each value must be a "
                             "multiple of 16. For debugging only.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Prove the script runs without queueing for a GPU: 2 patients per "
                             "split, 1 epoch, batch 1, 64x64x48 crop, CPU allowed. Takes about a "
                             "minute on the login node. Results are meaningless - this only "
                             "checks that the code works.")
    args = parser.parse_args()

    if args.smoke_test:
        args.limit = args.limit or 2
        args.epochs = 1
        args.batch_size = 1
        args.crop = args.crop or [64, 64, 48]
        args.workers = 0
        args.out_dir = os.path.join(args.out_dir, "smoke_test")
        print("SMOKE TEST: tiny run to check the code only. Do not report these numbers.\n")

    if args.crop and any(size % 16 for size in args.crop):
        sys.exit(f"ERROR: --crop values must each be a multiple of 16, got {args.crop}. "
                 "The U-Net downsamples by 2 four times.")

    monai.utils.set_determinism(seed=args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu = torch.cuda.get_device_properties(0)
        print(f"torch {torch.__version__} (CUDA build {torch.version.cuda}) "
              f"| MONAI {monai.__version__}")
        print(f"GPU: {gpu.name} | {gpu.total_memory / 1e9:.1f} GB")
    elif args.smoke_test:
        device = torch.device("cpu")
        print(f"torch {torch.__version__} | MONAI {monai.__version__}")
        print("No GPU visible - running the smoke test on CPU.")
    else:
        sys.exit("ERROR: no GPU visible. Submit a real run with sbatch on a GPU partition, "
                 "or add --smoke-test to check the code on CPU.")

    use_amp = not args.no_amp and device.type == "cuda"
    print(f"batch size {args.batch_size} | lr {args.lr} | epochs {args.epochs} "
          f"| mixed precision {use_amp} | seed {args.seed}"
          + (f" | crop {tuple(args.crop)}" if args.crop else ""))

    # --- Data -------------------------------------------------------------
    train_files = load_split(args.cache_dir, "train", args.limit,
                             args.fold, args.folds_file)
    val_files = load_split(args.cache_dir, "val", args.limit,
                           args.fold, args.folds_file)
    print(f"train {len(train_files)} patients | val {len(val_files)} patients")

    train_loader = DataLoader(
        CachedBraTS(args.cache_dir, train_files, build_augmentation(), args.crop),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=device.type == "cuda", drop_last=len(train_files) > args.batch_size,
        persistent_workers=args.workers > 0)
    val_loader = DataLoader(
        CachedBraTS(args.cache_dir, val_files, None, args.crop),   # no augmentation, ever
        batch_size=1, shuffle=False, num_workers=args.workers,
        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)

    # --- Model ------------------------------------------------------------
    model = build_model(args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"3D U-Net: {n_params / 1e6:.1f}M parameters")

    loss_function = DiceCELoss(sigmoid=True, squared_pred=True)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    dice_metric = DiceMetric(include_background=True, reduction="mean")

    start_epoch, best_dice, history = 0, 0.0, []
    last_path = os.path.join(args.out_dir, "last.pt")
    best_path = os.path.join(args.out_dir, "best.pt")

    if args.resume and os.path.isfile(last_path):
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        best_dice = state["best_dice"]
        history = state["history"]
        for _ in range(start_epoch):
            schedule.step()
        print(f"Resumed from epoch {start_epoch} (best val Dice so far {best_dice:.4f})")

    # --- Training ---------------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_start, running_loss = time.time(), 0.0

        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["seg"].to(device, non_blocking=True)

            optimiser.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=use_amp):
                loss = loss_function(model(images), targets)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            running_loss += loss.item()

        schedule.step()
        train_loss = running_loss / max(len(train_loader), 1)
        val_dice = run_validation(model, val_loader, device, dice_metric, use_amp)
        elapsed = time.time() - epoch_start

        history.append({"epoch": epoch, "train_loss": train_loss, "val_dice": val_dice,
                        "lr": schedule.get_last_lr()[0], "seconds": round(elapsed, 1)})
        marker = ""
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "val_dice": val_dice, "args": vars(args)}, best_path)
            marker = "  <- best"

        print(f"epoch {epoch + 1:3d}/{args.epochs}  loss {train_loss:.4f}  "
              f"val Dice {val_dice:.4f}  {elapsed:5.1f}s{marker}", flush=True)

        save_checkpoint(last_path, epoch, model, optimiser, scaler, best_dice, history)
        pd.DataFrame(history).to_csv(os.path.join(args.out_dir, "history.csv"), index=False)

    # --- Summary ----------------------------------------------------------
    summary = {"finished": datetime.now().isoformat(timespec="seconds"),
               "epochs": args.epochs,
               "best_val_dice": best_dice,
               "parameters": n_params,
               "settings": vars(args),
               "augmentation": {"flip_prob_per_axis": FLIP_PROB,
                                "rotate90_prob": ROTATE90_PROB,
                                "scale_intensity": [SCALE_INTENSITY_PROB, SCALE_INTENSITY_FACTOR],
                                "shift_intensity": [SHIFT_INTENSITY_PROB, SHIFT_INTENSITY_OFFSET],
                                "gaussian_noise": [NOISE_PROB, NOISE_STD]},
               "torch_version": torch.__version__,
               "monai_version": monai.__version__}
    with open(os.path.join(args.out_dir, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nBest validation Dice: {best_dice:.4f}")
    print(f"Saved: {best_path}")


if __name__ == "__main__":
    main()
