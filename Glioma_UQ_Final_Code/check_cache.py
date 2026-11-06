"""Check the preprocessed files from prepare_data.py: shapes, NaNs, empty masks
and tumours touching the edge of the volume.

Usage: python check_cache.py --cache-dir /path/to/cache [--full]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

EXPECTED_IMAGE_SHAPE = (4, 128, 128, 96)
BRATS_LABELS = {0, 1, 2, 4}


def tumour_touches_edge(seg):
    """True if any tumour voxel sits on the outer surface of the volume."""
    return bool(seg[0].any() or seg[-1].any() or
                seg[:, 0].any() or seg[:, -1].any() or
                seg[:, :, 0].any() or seg[:, :, -1].any())


def main():
    parser = argparse.ArgumentParser(description="Check the preprocessed cache.")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--full", action="store_true",
                        help="Also load the images to check shape/NaN (slower, reads every file)")
    args = parser.parse_args()

    manifest_path = os.path.join(args.cache_dir, "manifest.csv")
    if not os.path.isfile(manifest_path):
        sys.exit(f"ERROR: {manifest_path} not found. Run prepare_data.py first.")
    manifest = pd.read_csv(manifest_path)

    problems, at_edge = [], []
    for _, row in manifest.iterrows():
        path = os.path.join(args.cache_dir, row["file"])
        with np.load(path) as saved:
            seg = saved["seg"]

            labels = set(np.unique(seg).tolist())
            if not labels <= BRATS_LABELS:
                problems.append(f"{row['subject_id']}: unexpected mask labels {sorted(labels)}")
            if not (seg > 0).any():
                problems.append(f"{row['subject_id']}: empty tumour mask")
            if tumour_touches_edge(seg):
                at_edge.append(row["subject_id"])

            if args.full:
                image = saved["image"]
                if image.shape != EXPECTED_IMAGE_SHAPE:
                    problems.append(f"{row['subject_id']}: image shape {image.shape}")
                if not np.isfinite(image).all():
                    problems.append(f"{row['subject_id']}: image contains NaN or infinity")

    print(f"Checked {len(manifest)} patients in {args.cache_dir}")
    print(f"Trimmed brains (per manifest): {int(manifest['brain_was_cropped'].sum())}")
    print(f"Tumours touching the volume edge: {len(at_edge)}")
    if at_edge:
        print("  These tumours may have been clipped by the fixed-size crop:")
        for subject_id in at_edge:
            print("   -", subject_id)

    tumour = manifest["tumour_voxels"]
    print(f"\nTumour size (voxels): min {tumour.min()}, median {int(tumour.median())}, max {tumour.max()}")
    print("\nPatients per split:")
    print(manifest.groupby(["split", "grade"]).size().to_string())

    if problems:
        print(f"\nPROBLEMS ({len(problems)}):")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("\nNo problems found.")


if __name__ == "__main__":
    main()
