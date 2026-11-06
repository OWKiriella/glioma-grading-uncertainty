"""Preprocess every patient once and save one .npz per patient.

Steps: RAS orientation, crop to the brain, resample to 1.5 mm,
pad/crop to 128 x 128 x 96, z-score each scan on brain voxels only.
Saved arrays: image [4, 128, 128, 96] (t1, t1ce, t2, flair) and seg [128, 128, 96].
Augmentation is done later, during training.

Usage: python prepare_data.py --data-dir BRATS_DIR --out-dir CACHE
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import monai
import numpy as np
import pandas as pd
import torch
from monai.transforms import (Compose, CropForegroundd, EnsureChannelFirstd, LoadImaged,
                              NormalizeIntensityd, Orientationd, ResizeWithPadOrCropd, Spacingd)

MODALITIES = ["t1", "t1ce", "t2", "flair"]
VOXEL_SIZE_MM = (1.5, 1.5, 1.5)
FINAL_SHAPE = (128, 128, 96)
BRATS_LABELS = {0, 1, 2, 4}


def build_transforms():
    """Two parts, so the brain size can be recorded before the final pad/crop."""
    keys = ["image", "seg"]

    load_and_resample = Compose([
        LoadImaged(keys=keys, image_only=True),        # 4 scan files are stacked into 4 channels
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        CropForegroundd(keys=keys, source_key="image", allow_smaller=True),
        Spacingd(keys=keys, pixdim=VOXEL_SIZE_MM, mode=("bilinear", "nearest")),
    ])

    to_final_shape = Compose([
        ResizeWithPadOrCropd(keys=keys, spatial_size=FINAL_SHAPE),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])

    return load_and_resample, to_final_shape


def process_patient(row, data_dir, out_path):
    """Preprocess and save one patient. Returns the values for manifest.csv."""
    try:
        return _process_patient(row, data_dir, out_path)
    except Exception as error:
        raise RuntimeError(describe(error)) from None  # a short, readable reason for the log


def _process_patient(row, data_dir, out_path):
    torch.set_num_threads(1)  # each worker process uses one CPU core
    load_and_resample, to_final_shape = build_transforms()

    data = {"image": [os.path.join(data_dir, row[m]) for m in MODALITIES],
            "seg": os.path.join(data_dir, row["seg"])}

    data = load_and_resample(data)
    brain_shape = tuple(int(s) for s in data["image"].shape[1:])
    data = to_final_shape(data)

    image = np.asarray(data["image"], dtype=np.float32)
    seg = np.rint(np.asarray(data["seg"])[0]).astype(np.uint8)

    labels = set(np.unique(seg).tolist())
    if not labels <= BRATS_LABELS:
        raise ValueError(f"unexpected mask labels {sorted(labels)}")

    # Save under a temporary name first, so a job killed mid-save never leaves a half-written file
    tmp_path = out_path + ".tmp.npz"
    np.savez(tmp_path, image=image, seg=seg,
             brain_shape=np.array(brain_shape), tumour_voxels=np.array((seg > 0).sum()))
    os.replace(tmp_path, out_path)
    return brain_shape, int((seg > 0).sum())


def describe(error):
    """Short error message: the original cause, not MONAI's wrapper."""
    while error.__cause__ is not None:
        error = error.__cause__
    return f"{type(error).__name__}: {error}"


def read_checks(out_path):
    """Read the check values back from a file that is already done."""
    with np.load(out_path) as saved:
        return tuple(int(s) for s in saved["brain_shape"]), int(saved["tumour_voxels"])


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Preprocess all patients once and cache the result.")
    parser.add_argument("--data-dir", required=True,
                        help="Folder containing the BraTS20_Training_XXX folders")
    parser.add_argument("--out-dir", required=True,
                        help="Where to save the preprocessed files (use scratch on the HPC)")
    parser.add_argument("--splits", default=os.path.join(here, "splits.csv"),
                        help="Output of make_splits.py (default: splits.csv next to this script)")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)),
                        help="Patients processed in parallel (default: CPUs given by SLURM, else 1)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Redo patients that were already processed")
    args = parser.parse_args()

    if not os.path.isfile(args.splits):
        sys.exit(f"ERROR: {args.splits} not found. Run make_splits.py first.")
    patients = pd.read_csv(args.splits)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Patients: {len(patients)} | data: {args.data_dir}")
    print(f"Output: {args.out_dir} | workers: {args.workers}")

    results, failed = {}, {}
    todo = []
    for _, row in patients.iterrows():
        out_path = os.path.join(args.out_dir, f"{row['subject_id']}.npz")
        if os.path.exists(out_path) and not args.overwrite:
            try:
                results[row["subject_id"]] = read_checks(out_path)  # done in an earlier run
                continue
            except Exception:
                print(f"  {row['subject_id']}: saved file is unreadable, redoing it")
        todo.append((row, out_path))
    if results:
        print(f"Already done (skipped): {len(results)}")

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(process_patient, row, args.data_dir, out_path): row["subject_id"]
                for row, out_path in todo}
        for i, job in enumerate(as_completed(jobs), start=1):
            subject_id = jobs[job]
            try:
                results[subject_id] = job.result()
            except Exception as error:
                failed[subject_id] = str(error)
                print(f"  FAILED {subject_id}: {failed[subject_id]}")
            if i % 25 == 0 or i == len(todo):
                print(f"  processed {i}/{len(todo)}", flush=True)

    # manifest.csv: one row per successfully processed patient
    rows = []
    for _, row in patients.iterrows():
        if row["subject_id"] not in results:
            continue
        brain_shape, tumour_voxels = results[row["subject_id"]]
        rows.append({"subject_id": row["subject_id"],
                     "split": row["split"],
                     "label": row["label"],
                     "grade": row["grade"],
                     "file": f"{row['subject_id']}.npz",
                     "brain_shape": "x".join(map(str, brain_shape)),
                     "brain_was_cropped": any(b > f for b, f in zip(brain_shape, FINAL_SHAPE)),
                     "tumour_voxels": tumour_voxels})
    manifest = pd.DataFrame(rows)
    manifest.to_csv(os.path.join(args.out_dir, "manifest.csv"), index=False)

    config = {"created": datetime.now().isoformat(timespec="seconds"),
              "modalities_channel_order": MODALITIES,
              "orientation": "RAS",
              "background_crop": "bounding box of non-zero voxels (any modality)",
              "voxel_size_mm": VOXEL_SIZE_MM,
              "final_shape": FINAL_SHAPE,
              "image_interpolation": "bilinear",
              "mask_interpolation": "nearest",
              "normalisation": "z-score per modality, brain (non-zero) voxels only",
              "image_dtype": "float32",
              "monai_version": monai.__version__,
              "torch_version": torch.__version__}
    with open(os.path.join(args.out_dir, "preprocess_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Summary
    print(f"\nDone: {len(manifest)} patients saved to {args.out_dir}")
    if len(manifest):
        print(f"Brains larger than {FINAL_SHAPE} (edges trimmed): {int(manifest['brain_was_cropped'].sum())}")
        no_tumour = manifest.loc[manifest["tumour_voxels"] == 0, "subject_id"].tolist()
        if no_tumour:
            print(f"WARNING: empty tumour mask after preprocessing: {no_tumour}")
    if failed:
        print(f"\nERROR: {len(failed)} patients failed:")
        for subject_id, error in failed.items():
            print(f"  - {subject_id}: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
