"""Train the grading CNN (HGG = 1, LGG = 0).

    --variant whole-brain   4 channels: t1, t1ce, t2, flair
    --variant mask-guided   the same plus the PREDICTED tumour mask (5 channels)

The loss is weighted by inverse class frequency and the best epoch is chosen
by validation balanced accuracy. Run with --seed 1 to 5 for the deep ensemble.

Usage: python train_grader.py --cache-dir CACHE --out-dir RUNS/grader_whole-brain/seed1 \
                              --variant whole-brain --seed 1
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
import torch.nn as nn
from monai.data import DataLoader, Dataset
from monai.transforms import (CenterSpatialCropd, Compose, ConcatItemsd, DeleteItemsd,
                              RandFlipd, RandGaussianNoised, RandRotate90d,
                              RandScaleIntensityd, RandShiftIntensityd)
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score)

# --- Augmentation settings (the same as train_seg.py) ----------------------
FLIP_PROB = 0.5           # per axis, all three axes
ROTATE90_PROB = 0.5       # axial plane only
SCALE_INTENSITY_PROB, SCALE_INTENSITY_FACTOR = 0.5, 0.1
SHIFT_INTENSITY_PROB, SHIFT_INTENSITY_OFFSET = 0.5, 0.1
NOISE_PROB, NOISE_STD = 0.15, 0.1

WIDTHS = (32, 64, 128, 256)   # channels of the four convolution blocks

DROPOUT = 0.25            # Mosquera Rojas et al., Melba 2026:033: for the tumour grade
                          # task ECE was lowest at 0.25, and 0.25 was their selected
                          # configuration (T = 30, delta = 0.25). Verified in the paper.
GRADE_NAMES = {0: "LGG", 1: "HGG"}
VARIANT_CHANNELS = {"whole-brain": 4, "mask-guided": 5}


class CachedGrading(Dataset):
    """One .npz per patient, returning the image and the grade label.
    For mask-guided, the predicted mask is added as a fifth channel.
    """

    def __init__(self, cache_dir, records, transform=None, crop=None, mask_dir=None):
        self.cache_dir = cache_dir
        self.records = list(records)          # (filename, label) pairs
        self.transform = transform
        self.crop = crop                      # smoke-test only
        self.mask_dir = mask_dir               # None for the whole-brain variant

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        filename, label = self.records[index]
        with np.load(os.path.join(self.cache_dir, filename)) as saved:
            image = saved["image"].astype(np.float32)           # [4, 128, 128, 96]

        sample = {"image": torch.from_numpy(image),
                  "label": torch.tensor(label, dtype=torch.long)}

        if self.mask_dir is not None:
            mask_path = os.path.join(self.mask_dir, filename)
            if not os.path.isfile(mask_path):
                raise FileNotFoundError(
                    f"No predicted mask for {filename} in {self.mask_dir}. "
                    "Run the Stage 1 model over every split first.")
            with np.load(mask_path) as saved:
                mask = (saved["mask"] > 0).astype(np.float32)[None]
            # Kept separate so intensity augmentation does not change it.
            sample["mask"] = torch.from_numpy(mask)

        keys = ["image", "mask"] if "mask" in sample else ["image"]
        if self.crop is not None:
            sample = CenterSpatialCropd(keys=keys, roi_size=self.crop)(sample)
        if self.transform is not None:
            sample = self.transform(sample)
        elif "mask" in sample:
            # No augmentation - validation and inference - so join them here.
            sample["image"] = torch.cat([sample["image"], sample.pop("mask")], dim=0)
        return sample


def build_augmentation(with_mask=False):
    """Training-only augmentation. Flips and rotations are applied to the scans
    and the mask together; intensity changes only to the scans, so the mask
    stays binary.
    """
    spatial = ["image", "mask"] if with_mask else ["image"]
    steps = [
        RandFlipd(keys=spatial, prob=FLIP_PROB, spatial_axis=0),
        RandFlipd(keys=spatial, prob=FLIP_PROB, spatial_axis=1),
        RandFlipd(keys=spatial, prob=FLIP_PROB, spatial_axis=2),
        RandRotate90d(keys=spatial, prob=ROTATE90_PROB, spatial_axes=(0, 1)),
        RandScaleIntensityd(keys="image", factors=SCALE_INTENSITY_FACTOR,
                            prob=SCALE_INTENSITY_PROB, channel_wise=True),
        RandShiftIntensityd(keys="image", offsets=SHIFT_INTENSITY_OFFSET,
                            prob=SHIFT_INTENSITY_PROB, channel_wise=True),
        RandGaussianNoised(keys="image", prob=NOISE_PROB, std=NOISE_STD),
    ]
    if with_mask:
        steps += [ConcatItemsd(keys=["image", "mask"], name="image", dim=0),
                  DeleteItemsd(keys="mask")]
    return Compose(steps)


class Simple3DCNN(nn.Module):
    """Small 3D CNN: four convolution blocks, global average pooling and a
    linear layer. Dropout in the last two blocks and before the classifier is
    what Monte Carlo dropout uses at test time.
    """

    def __init__(self, in_channels=4, n_classes=2, dropout=DROPOUT,
                 widths=WIDTHS):
        super().__init__()
        blocks, channels_in = [], in_channels
        for position, channels_out in enumerate(widths):
            blocks.append(self._block(channels_in, channels_out,
                                      dropout if position >= 2 else 0.0))
            channels_in = channels_out
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Dropout(dropout),
                                        nn.Linear(channels_in, n_classes))

    @staticmethod
    def _block(channels_in, channels_out, dropout):
        layers = [
            nn.Conv3d(channels_in, channels_out, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(channels_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels_out, channels_out, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(channels_out, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        ]
        if dropout > 0:
            layers.append(nn.Dropout3d(dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_model(in_channels, dropout):
    """The grader, shared with predict_mc.py."""
    return Simple3DCNN(in_channels=in_channels, n_classes=2, dropout=dropout)


def load_split(cache_dir, split, limit=0, fold=None, folds_file=None):
    """(filename, label) pairs for one split."""
    rows = folds.resolve_split(cache_dir, split, fold, folds_file, limit)
    print("  " + folds.describe(split, fold, rows))
    return list(zip(rows["file"], rows["label"].astype(int)))


def class_weights(records, device):
    """Inverse class frequency on the training split: n / (2 * n_class)."""
    labels = np.array([label for _, label in records])
    counts = np.array([(labels == 0).sum(), (labels == 1).sum()], dtype=np.float64)
    if counts.min() == 0:
        print("WARNING: one grade is missing from this split, using equal weights.")
        return torch.ones(2, device=device), counts.astype(int)
    weights = labels.size / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device), counts.astype(int)


def run_validation(model, loader, device, use_amp, loss_function=None):
    """Validation loss, balanced accuracy, accuracy, F1, sensitivity,
    specificity, confusion matrix and AUC.
    """
    model.eval()
    true_labels, predicted_labels, hgg_probabilities = [], [], []
    total_loss, batches = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=use_amp):
                logits = model(images)
            if loss_function is not None:
                labels = batch["label"].to(device, non_blocking=True)
                total_loss += loss_function(logits.float(), labels).item()
                batches += 1
            probabilities = torch.softmax(logits.float(), dim=1)
            true_labels.extend(batch["label"].tolist())
            predicted_labels.extend(probabilities.argmax(dim=1).cpu().tolist())
            hgg_probabilities.extend(probabilities[:, 1].cpu().tolist())

    metrics = {"loss": total_loss / batches if batches else float("nan"),
               "balanced_accuracy": balanced_accuracy_score(true_labels, predicted_labels),
               "accuracy": float(np.mean(np.array(true_labels) == np.array(predicted_labels))),
               "f1_hgg": f1_score(true_labels, predicted_labels, pos_label=1, zero_division=0)}

    # Sensitivity and specificity need both grades present in the split.
    matrix = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])
    (true_lgg, false_hgg), (false_lgg, true_hgg) = matrix
    metrics["sensitivity_hgg"] = true_hgg / (true_hgg + false_lgg) if (true_hgg + false_lgg) else float("nan")
    metrics["specificity_lgg"] = true_lgg / (true_lgg + false_hgg) if (true_lgg + false_hgg) else float("nan")
    metrics["confusion"] = matrix.tolist()
    try:
        metrics["auc"] = roc_auc_score(true_labels, hgg_probabilities)
    except ValueError:
        metrics["auc"] = float("nan")       # only one grade present (smoke test)
    return metrics


def save_checkpoint(path, epoch, model, optimiser, scaler, best_score, history):
    tmp_path = path + ".tmp"          # never leave a half-written checkpoint
    torch.save({"epoch": epoch,
                "model": model.state_dict(),
                "optimiser": optimiser.state_dict(),
                "scaler": scaler.state_dict(),
                "best_score": best_score,
                "history": history}, tmp_path)
    os.replace(tmp_path, path)


def main():
    parser = argparse.ArgumentParser(description="Train the Stage 2 HGG/LGG grading model.")
    parser.add_argument("--cache-dir", required=True, help="Folder of .npz files from prepare_data.py")
    parser.add_argument("--out-dir", required=True, help="Where to write checkpoints and the log")
    parser.add_argument("--variant", choices=sorted(VARIANT_CHANNELS), default="whole-brain",
                        help="whole-brain = 4 MRI channels only (needs no segmentation). "
                             "mask-guided = plus the predicted tumour mask as a 5th channel.")
    parser.add_argument("--mask-dir", default=None,
                        help="Folder of PREDICTED masks, required by --variant mask-guided")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Patients per training step (fits a 48 GB L40S)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=DROPOUT,
                        help="Kept active at test time by predict_mc.py - do not set to 0")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    parser.add_argument("--seed", type=int, default=42,
                        help="Change this (1..5) to train the members of the deep ensemble")
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
                        help="Prove the script runs without queueing for a GPU: 4 patients per "
                             "split, 1 epoch, batch 2, 64x64x48 crop, CPU allowed. Takes about a "
                             "minute on the login node. Results are meaningless - this only "
                             "checks that the code works.")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="Train on CPU at full size. For when both GPU nodes are held by "
                             "long jobs and the cpu partition is idle: the run is real, not a "
                             "smoke test, but it is many times slower. Mixed precision is off "
                             "on CPU, so results differ from a GPU run in the last few decimal "
                             "places - keep a comparison on one device or the other.")
    args = parser.parse_args()

    if args.smoke_test:
        args.limit = args.limit or 4
        args.epochs = 1
        args.batch_size = 2
        args.crop = args.crop or [64, 64, 48]
        args.workers = 0
        args.out_dir = os.path.join(args.out_dir, "smoke_test")
        print("SMOKE TEST: tiny run to check the code only. Do not report these numbers.\n")

    if args.crop and any(size % 16 for size in args.crop):
        sys.exit(f"ERROR: --crop values must each be a multiple of 16, got {args.crop}. "
                 "The network halves each dimension four times.")

    in_channels = VARIANT_CHANNELS[args.variant]
    if args.variant == "mask-guided" and not args.mask_dir:
        sys.exit("ERROR: --variant mask-guided needs --mask-dir pointing at PREDICTED masks "
                 "from the Stage 1 model. Never pass the BraTS ground-truth masks here.")
    if args.mask_dir and args.variant == "whole-brain":
        sys.exit("ERROR: --mask-dir given but --variant is whole-brain. Pick one.")

    monai.utils.set_determinism(seed=args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu = torch.cuda.get_device_properties(0)
        print(f"torch {torch.__version__} (CUDA build {torch.version.cuda}) "
              f"| MONAI {monai.__version__}")
        print(f"GPU: {gpu.name} | {gpu.total_memory / 1e9:.1f} GB")
    elif args.smoke_test or args.allow_cpu:
        device = torch.device("cpu")
        print(f"torch {torch.__version__} | MONAI {monai.__version__}")
        reason = "smoke test" if args.smoke_test else "--allow-cpu given"
        print(f"No GPU visible - running on CPU ({reason}).")
        if args.allow_cpu and not args.smoke_test:
            print("This is a full-size run on CPU. Expect it to take far longer than on a GPU, "
                  "and do not compare it against GPU-trained runs.")
    else:
        sys.exit("ERROR: no GPU visible. Submit a real run with sbatch on a GPU partition, "
                 "add --smoke-test to check the code on CPU, or --allow-cpu to train on CPU "
                 "at full size.")

    use_amp = not args.no_amp and device.type == "cuda"
    print(f"variant {args.variant} ({in_channels} channels) | dropout {args.dropout}")
    print(f"batch size {args.batch_size} | lr {args.lr} | weight decay {args.weight_decay} "
          f"| epochs {args.epochs} | mixed precision {use_amp} | seed {args.seed}"
          + (f" | crop {tuple(args.crop)}" if args.crop else ""))

    # --- Data -------------------------------------------------------------
    train_records = load_split(args.cache_dir, "train", args.limit,
                               args.fold, args.folds_file)
    val_records = load_split(args.cache_dir, "val", args.limit,
                             args.fold, args.folds_file)

    weights, counts = class_weights(train_records, device)
    print(f"train {len(train_records)} patients ({counts[1]} HGG / {counts[0]} LGG) "
          f"| val {len(val_records)} patients")
    print(f"loss weights: LGG {weights[0]:.3f}, HGG {weights[1]:.3f}")

    train_loader = DataLoader(
        CachedGrading(args.cache_dir, train_records,
                      build_augmentation(with_mask=args.mask_dir is not None),
                      args.crop, args.mask_dir),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=len(train_records) > args.batch_size,
        persistent_workers=args.workers > 0)
    val_loader = DataLoader(
        CachedGrading(args.cache_dir, val_records, None, args.crop, args.mask_dir),
        batch_size=1, shuffle=False, num_workers=args.workers,
        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)

    # --- Model ------------------------------------------------------------
    model = build_model(in_channels, args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Simple3DCNN: {n_params / 1e6:.1f}M parameters")

    loss_function = nn.CrossEntropyLoss(weight=weights)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best_score, history = 0, 0.0, []
    last_path = os.path.join(args.out_dir, "last.pt")
    best_path = os.path.join(args.out_dir, "best.pt")

    if args.resume and os.path.isfile(last_path):
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        best_score = state["best_score"]
        history = state["history"]
        for _ in range(start_epoch):
            schedule.step()
        print(f"Resumed from epoch {start_epoch} "
              f"(best val balanced accuracy so far {best_score:.4f})")

    # --- Training ---------------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_start, running_loss = time.time(), 0.0

        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimiser.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=use_amp):
                loss = loss_function(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            running_loss += loss.item()

        schedule.step()
        train_loss = running_loss / max(len(train_loader), 1)
        metrics = run_validation(model, val_loader, device, use_amp, loss_function)
        elapsed = time.time() - epoch_start

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": metrics["loss"],
                        "val_balanced_accuracy": metrics["balanced_accuracy"],
                        "val_accuracy": metrics["accuracy"],
                        "val_sensitivity_hgg": metrics["sensitivity_hgg"],
                        "val_specificity_lgg": metrics["specificity_lgg"],
                        "val_f1_hgg": metrics["f1_hgg"],
                        "val_auc": metrics["auc"],
                        "lr": schedule.get_last_lr()[0],
                        "seconds": round(elapsed, 1)})

        marker = ""
        if metrics["balanced_accuracy"] > best_score:
            best_score = metrics["balanced_accuracy"]
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "metrics": metrics, "args": vars(args)}, best_path)
            marker = "  <- best"

        print(f"epoch {epoch + 1:3d}/{args.epochs}  loss {train_loss:.4f}  "
              f"val loss {metrics['loss']:.4f}  "
              f"val bal.acc {metrics['balanced_accuracy']:.4f}  "
              f"(sens {metrics['sensitivity_hgg']:.2f} spec {metrics['specificity_lgg']:.2f} "
              f"AUC {metrics['auc']:.3f})  {elapsed:5.1f}s{marker}", flush=True)

        save_checkpoint(last_path, epoch, model, optimiser, scaler, best_score, history)
        pd.DataFrame(history).to_csv(os.path.join(args.out_dir, "history.csv"), index=False)

    # --- Summary ----------------------------------------------------------
    best_val_loss = min(record["val_loss"] for record in history) if history else None
    summary = {"finished": datetime.now().isoformat(timespec="seconds"),
               "fold": args.fold,
               "variant": args.variant,
               "in_channels": in_channels,
               "widths": list(WIDTHS),
               "epochs": args.epochs,
               "best_val_balanced_accuracy": best_score,
               "best_val_loss": best_val_loss,
               "parameters": n_params,
               "train_counts": {"LGG": int(counts[0]), "HGG": int(counts[1])},
               "loss_weights": {"LGG": float(weights[0]), "HGG": float(weights[1])},
               "settings": vars(args),
               "augmentation": {"flip_prob_per_axis": FLIP_PROB,
                                "rotate90_prob": ROTATE90_PROB,
                                "scale_intensity": [SCALE_INTENSITY_PROB, SCALE_INTENSITY_FACTOR],
                                "shift_intensity": [SHIFT_INTENSITY_PROB, SHIFT_INTENSITY_OFFSET],
                                "gaussian_noise": [NOISE_PROB, NOISE_STD],
                                "intensity_on_mri_channels_only": True},
               "torch_version": torch.__version__,
               "monai_version": monai.__version__}
    with open(os.path.join(args.out_dir, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nBest validation balanced accuracy: {best_score:.4f}")
    print("A model that ignored the scan and always answered HGG would score 0.5000.")
    print(f"Saved: {best_path}")


if __name__ == "__main__":
    main()
