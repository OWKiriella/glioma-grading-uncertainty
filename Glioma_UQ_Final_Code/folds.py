"""Shared helper: which patients belong to train / val / test.

Without --fold the split comes from manifest.csv (259 / 55 / 55).
With --fold k, fold k of splits_cv.csv is the validation set and the other
folds are the training set. The test set is never part of any fold.
"""
import os
import sys

import pandas as pd

TEST_FOLD = -1


def _load_manifest(cache_dir):
    path = os.path.join(cache_dir, "manifest.csv")
    if not os.path.isfile(path):
        sys.exit(f"ERROR: {path} not found. Run prepare_data.py first.")
    return pd.read_csv(path)


def _load_folds(folds_file):
    if not os.path.isfile(folds_file):
        sys.exit(f"ERROR: {folds_file} not found. Run make_folds.py first.")
    folds = pd.read_csv(folds_file)
    if "fold" not in folds.columns:
        sys.exit(f"ERROR: {folds_file} has no 'fold' column. Run make_folds.py.")
    return folds[["subject_id", "fold"]]


def resolve_split(cache_dir, split, fold=None, folds_file=None, limit=0):
    """Manifest rows for one split, as a DataFrame."""
    manifest = _load_manifest(cache_dir)

    if fold is None:
        rows = manifest.loc[manifest["split"] == split]
        if rows.empty:
            sys.exit(f"ERROR: no patients for split '{split}' in {cache_dir}/manifest.csv")
        return rows.head(limit) if limit else rows

    if fold == TEST_FOLD or fold < 0:
        sys.exit(f"ERROR: --fold {fold} is not a fold. The test split is reached with "
                 "--split test, never through a fold number.")

    folds = _load_folds(folds_file)
    merged = manifest.merge(folds, on="subject_id", how="left", validate="one_to_one")

    missing = merged["fold"].isna().sum()
    if missing:
        sys.exit(f"ERROR: {missing} cached patients are absent from {folds_file}. "
                 "The fold file and the cache describe different patient sets.")
    merged["fold"] = merged["fold"].astype(int)

    if fold not in set(merged.loc[merged["fold"] != TEST_FOLD, "fold"]):
        available = sorted(set(merged.loc[merged["fold"] != TEST_FOLD, "fold"]))
        sys.exit(f"ERROR: fold {fold} not found in {folds_file}. Available: {available}")

    if split == "test":
        rows = merged.loc[merged["fold"] == TEST_FOLD]
    elif split == "val":
        rows = merged.loc[merged["fold"] == fold]
    elif split == "train":
        # Everything that is neither held out nor sealed.
        rows = merged.loc[(merged["fold"] != fold) & (merged["fold"] != TEST_FOLD)]
    else:
        sys.exit(f"ERROR: unknown split '{split}'")

    if rows.empty:
        sys.exit(f"ERROR: no patients for split '{split}' at fold {fold}")

    # A held-out patient reaching the training set would invalidate the fold.
    if split == "train":
        held_out = set(merged.loc[merged["fold"] == fold, "subject_id"])
        sealed = set(merged.loc[merged["fold"] == TEST_FOLD, "subject_id"])
        overlap = set(rows["subject_id"]) & (held_out | sealed)
        if overlap:
            sys.exit(f"ERROR: {len(overlap)} held-out or test patients reached the "
                     f"training set at fold {fold}. Refusing to train.")

    return rows.head(limit) if limit else rows


def describe(split, fold, rows):
    """One line with the patient counts, for the log."""
    where = "manifest" if fold is None else f"fold {fold}"
    counts = ""
    if "label" in rows.columns:
        hgg = int((rows["label"] == 1).sum())
        counts = f" ({hgg} HGG / {len(rows) - hgg} LGG)"
    return f"{split:5s} from {where}: {len(rows)} patients{counts}"


def add_fold_arguments(parser, here):
    """Add --fold and --folds-file to a parser."""
    parser.add_argument("--fold", type=int, default=None,
                        help="Cross-validation fold to hold out. Omit for the original "
                             "train/val split recorded in manifest.csv.")
    parser.add_argument("--folds-file", default=os.path.join(here, "splits_cv.csv"),
                        help="Fold assignments from make_folds.py (used only with --fold)")

