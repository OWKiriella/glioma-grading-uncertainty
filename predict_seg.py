"""Run the trained U-Net over every patient and save the predicted masks
(one .npz per patient, array "mask"), for the mask-guided grader.

The test Dice is hidden unless --report-test-dice is given.

Usage: python predict_seg.py --cache-dir CACHE --checkpoint RUNS/seg/best.pt \
                             --out-dir predicted_masks
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
from monai.metrics import DiceMetric

from train_seg import build_model

SPLITS = ["train", "val", "test"]


class CachedForPrediction(Dataset):
    """Image, ground-truth mask (for the Dice) and file name."""

    def __init__(self, cache_dir, files):
        self.cache_dir = cache_dir
        self.files = list(files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        filename = self.files[index]
        with np.load(os.path.join(self.cache_dir, filename)) as saved:
            image = saved["image"].astype(np.float32)          # [4, 128, 128, 96]
            seg = (saved["seg"] > 0).astype(np.float32)[None]  # [1, 128, 128, 96]
        return {"image": torch.from_numpy(image),
                "seg": torch.from_numpy(seg),
                "file": filename}


def load_manifest(cache_dir):
    manifest_path = os.path.join(cache_dir, "manifest.csv")
    if not os.path.isfile(manifest_path):
        sys.exit(f"ERROR: {manifest_path} not found. Run prepare_data.py first.")
    return pd.read_csv(manifest_path)


def main():
    parser = argparse.ArgumentParser(description="Predict whole-tumour masks with the Stage 1 model.")
    parser.add_argument("--cache-dir", required=True, help="Folder of .npz files from prepare_data.py")
    parser.add_argument("--checkpoint", required=True, help="best.pt written by train_seg.py")
    parser.add_argument("--out-dir", required=True, help="Where to write one mask .npz per patient")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability above which a voxel is called tumour")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    folds.add_fold_arguments(parser, os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="First N patients per split (0 = all)")
    parser.add_argument("--report-test-dice", action="store_true",
                        help="Also print the test-split Dice. Leave this off until the final "
                             "evaluation - the test set is meant to be looked at once.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="2 patients per split on CPU, written to a smoke_test subfolder. "
                             "Checks the code runs; the masks it makes are not usable.")
    args = parser.parse_args()

    if args.smoke_test:
        args.limit = args.limit or 2
        args.batch_size = 1
        args.workers = 0
        args.out_dir = os.path.join(args.out_dir, "smoke_test")
        print("SMOKE TEST: tiny run to check the code only. Do not use these masks.\n")

    if not os.path.isfile(args.checkpoint):
        sys.exit(f"ERROR: checkpoint not found: {args.checkpoint}\n"
                 "Train the segmentation model first (train_seg.py).")
    os.makedirs(args.out_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu = torch.cuda.get_device_properties(0)
        print(f"torch {torch.__version__} (CUDA build {torch.version.cuda}) | MONAI {monai.__version__}")
        print(f"GPU: {gpu.name} | {gpu.total_memory / 1e9:.1f} GB")
    elif args.smoke_test:
        device = torch.device("cpu")
        print(f"torch {torch.__version__} | MONAI {monai.__version__}")
        print("No GPU visible - running the smoke test on CPU.")
    else:
        sys.exit("ERROR: no GPU visible. Submit with sbatch, or add --smoke-test to check the code.")

    use_amp = not args.no_amp and device.type == "cuda"

    # --- Model (same dropout as in training, or the weights will not load) ---
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    trained_with = state.get("args", {})
    model = build_model(trained_with.get("dropout", 0.0)).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"Loaded {args.checkpoint} (epoch {state.get('epoch', '?')}, "
          f"val Dice {state.get('val_dice', float('nan')):.4f})")

    manifest = load_manifest(args.cache_dir)
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    per_patient, summary = [], {}

    # Without --fold: all three splits. With --fold k: only the patients held out
    # at fold k, so every mask comes from a model that did not train on it.
    if args.fold is None:
        groups = [(split, manifest.loc[manifest["split"] == split, "file"].tolist())
                  for split in SPLITS]
    else:
        held_out = folds.resolve_split(args.cache_dir, "val", args.fold, args.folds_file)
        print("  " + folds.describe("val", args.fold, held_out) + "  -> out-of-fold masks")
        groups = [(f"fold{args.fold}", held_out["file"].tolist())]

    for split, files in groups:
        if args.limit:
            files = files[:args.limit]
        if not files:
            print(f"{split}: no patients, skipped")
            continue

        loader = DataLoader(CachedForPrediction(args.cache_dir, files),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=device.type == "cuda")

        started = time.time()
        dice_metric.reset()
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device, non_blocking=True)
                targets = batch["seg"].to(device, non_blocking=True)
                with torch.autocast("cuda", enabled=use_amp):
                    logits = model(images)
                predicted = (torch.sigmoid(logits.float()) > args.threshold).float()
                dice_metric(y_pred=predicted, y=targets)

                masks = predicted[:, 0].to(torch.uint8).cpu().numpy()   # [B, 128, 128, 96]
                for mask, filename in zip(masks, batch["file"]):
                    path = os.path.join(args.out_dir, filename)
                    np.savez_compressed(path + ".tmp.npz", mask=mask)
                    os.replace(path + ".tmp.npz", path)
                    per_patient.append({"file": filename, "split": split,
                                        "predicted_voxels": int(mask.sum())})

        split_dice = float(dice_metric.aggregate().item())
        empty = sum(1 for row in per_patient
                    if row["split"] == split and row["predicted_voxels"] == 0)
        summary[split] = {"patients": len(files), "dice": split_dice, "empty_masks": empty}

        shown = f"Dice {split_dice:.4f}" if split != "test" or args.report_test_dice else "Dice hidden"
        print(f"{split:5s}  {len(files):3d} patients  {shown}  "
              f"empty masks {empty}  {time.time() - started:.1f}s", flush=True)

    # --- Save --------------------------------------------------------------
    table = pd.DataFrame(per_patient)
    table.to_csv(os.path.join(args.out_dir, "predicted_masks.csv"), index=False)

    saved = dict(summary)
    if not args.report_test_dice:
        saved.get("test", {}).pop("dice", None)   # keep the test score out of the file too
    with open(os.path.join(args.out_dir, "predict_summary.json"), "w") as f:
        json.dump({"finished": datetime.now().isoformat(timespec="seconds"),
                   "checkpoint": args.checkpoint,
                   "threshold": args.threshold,
                   "splits": saved,
                   "torch_version": torch.__version__,
                   "monai_version": monai.__version__}, f, indent=2)

    total_empty = sum(row["empty_masks"] for row in summary.values())
    print(f"\nWrote {len(table)} masks to {args.out_dir}")
    if total_empty:
        print(f"WARNING: {total_empty} patients got an all-zero mask. The mask-guided grader "
              "will see an empty fifth channel for them - mention this in the discussion.")


if __name__ == "__main__":
    main()
