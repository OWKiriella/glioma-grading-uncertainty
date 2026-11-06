"""Split the 314 non-test patients into 5 cross-validation folds, stratified
by grade. The test set is copied unchanged from splits.csv (fold = -1).

Usage: python make_folds.py [--folds 5 --seed 42]
"""
import argparse
import os
import sys

import pandas as pd
from sklearn.model_selection import StratifiedKFold

TEST_FOLD = -1


def assign_folds(splits, n_folds, seed):
    """Add a fold column: 0..n_folds-1 for non-test patients, -1 for test."""
    splits = splits.copy()
    splits["fold"] = TEST_FOLD

    pool = splits.loc[splits["split"] != "test"]
    if pool.empty:
        sys.exit("ERROR: no non-test patients found in the split file.")

    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (_, held_out) in enumerate(splitter.split(pool, pool["label"])):
        splits.loc[pool.index[held_out], "fold"] = fold
    return splits


def verify(original, folded, n_folds):
    """Check the folds before saving: the test set is unchanged and every
    non-test patient is in exactly one fold. Stops on any problem.
    """
    problems = []

    original_test = set(original.loc[original["split"] == "test", "subject_id"])
    folded_test = set(folded.loc[folded["fold"] == TEST_FOLD, "subject_id"])
    if original_test != folded_test:
        problems.append(f"test set changed: {len(original_test)} -> {len(folded_test)}, "
                        f"{len(original_test ^ folded_test)} patients differ")

    if len(folded) != len(original):
        problems.append(f"patient count changed: {len(original)} -> {len(folded)}")

    if folded["subject_id"].duplicated().any():
        problems.append("a subject_id appears more than once")

    pool = folded.loc[folded["fold"] != TEST_FOLD]
    assigned = sorted(pool["fold"].unique())
    if assigned != list(range(n_folds)):
        problems.append(f"folds present are {assigned}, expected {list(range(n_folds))}")

    expected_pool = set(original.loc[original["split"] != "test", "subject_id"])
    if set(pool["subject_id"]) != expected_pool:
        problems.append("the folded patients are not exactly the non-test patients")

    # A patient in two folds would leak between training and validation.
    if pool.groupby("subject_id")["fold"].nunique().max() > 1:
        problems.append("a patient is assigned to more than one fold")

    if problems:
        print("\nVERIFICATION FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(1)
    return True


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Add cross-validation folds to the existing split, leaving the test set alone.")
    parser.add_argument("--splits", default=os.path.join(here, "splits.csv"),
                        help="The existing split file to read (never modified)")
    parser.add_argument("--output", default=os.path.join(here, "splits_cv.csv"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42,
                        help="Same default as make_splits.py, so the folds are reproducible")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing splits_cv.csv")
    args = parser.parse_args()

    if not os.path.isfile(args.splits):
        sys.exit(f"ERROR: {args.splits} not found. Run make_splits.py first.")
    if os.path.exists(args.output) and not args.force:
        sys.exit(f"ERROR: {args.output} already exists. Models may already have been "
                 "trained against it.\nUse --force only if you really want to replace it.")

    original = pd.read_csv(args.splits)
    print(f"Read {len(original)} patients from {args.splits}")

    folded = assign_folds(original, args.folds, args.seed)
    verify(original, folded, args.folds)

    folded.to_csv(args.output, index=False)

    pool = folded.loc[folded["fold"] != TEST_FOLD]
    table = pd.crosstab(pool["fold"], pool["grade"], margins=True, margins_name="total")
    print(f"\nPatients per fold (the {len(pool)} non-test patients):")
    print(table.to_string())

    share = (pool.groupby("fold")["label"].mean() * 100).round(1)
    print(f"\nHGG share per fold (%): {share.to_dict()}")
    print(f"HGG share overall  (%): {pool['label'].mean() * 100:.1f}")

    held = folded.loc[folded["fold"] == TEST_FOLD]
    print(f"\nTest set untouched: {len(held)} patients, "
          f"{int(held['label'].sum())} HGG / {int((1 - held['label']).sum())} LGG")
    print("All verification checks passed.")
    print(f"\nSaved: {args.output}")
    print(f"{args.splits} was not modified.")


if __name__ == "__main__":
    main()
