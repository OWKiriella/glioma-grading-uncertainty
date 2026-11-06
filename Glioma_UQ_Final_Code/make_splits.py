"""Split the BraTS2020 patients 70 / 15 / 15 into train / validation / test,
stratified by grade, and save splits.csv. Run once; every later step reads it.

Usage: python make_splits.py --data-dir /path/to/MICCAI_BraTS2020_TrainingData
"""
import argparse
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

MODALITIES = ["t1", "t1ce", "t2", "flair"]
GRADE_TO_LABEL = {"LGG": 0, "HGG": 1}


def find_file(patient_dir, patient_id, suffix):
    """Return the file name for one scan, e.g. BraTS20_Training_001_t1.nii (or .nii.gz)."""
    for ext in (".nii", ".nii.gz"):
        name = f"{patient_id}_{suffix}{ext}"
        if os.path.isfile(os.path.join(patient_dir, name)):
            return name
    return None


def find_seg_file(patient_dir, patient_id):
    """Return the tumour-mask file name."""
    name = find_file(patient_dir, patient_id, "seg")
    if name is not None:
        return name

    candidates = [f for f in os.listdir(patient_dir)
                  if "seg" in f.lower() and f.endswith((".nii", ".nii.gz"))]
    if len(candidates) == 1:
        print(f"  NOTE: {patient_id} mask has a non-standard name: {candidates[0]}")
        return candidates[0]
    return None


def collect_patients(data_dir):
    """Read labels and check files. Stops with a clear error if anything is missing."""
    mapping_path = os.path.join(data_dir, "name_mapping.csv")
    if not os.path.isfile(mapping_path):
        sys.exit(f"ERROR: name_mapping.csv not found in {data_dir}")

    mapping = pd.read_csv(mapping_path)
    mapping = mapping[["BraTS_2020_subject_ID", "Grade"]].rename(
        columns={"BraTS_2020_subject_ID": "subject_id", "Grade": "grade"})

    if mapping["subject_id"].duplicated().any():
        sys.exit("ERROR: duplicate patient IDs in name_mapping.csv")

    unknown = set(mapping["grade"]) - set(GRADE_TO_LABEL)
    if unknown:
        sys.exit(f"ERROR: unexpected grade values in name_mapping.csv: {unknown}")

    rows, problems = [], []
    for patient_id, grade in zip(mapping["subject_id"], mapping["grade"]):
        patient_dir = os.path.join(data_dir, patient_id)
        if not os.path.isdir(patient_dir):
            problems.append(f"{patient_id}: folder missing")
            continue

        row = {"subject_id": patient_id,
               "grade": grade,
               "label": GRADE_TO_LABEL[grade]}

        for modality in MODALITIES:
            name = find_file(patient_dir, patient_id, modality)
            if name is None:
                problems.append(f"{patient_id}: {modality} scan missing")
            row[modality] = os.path.join(patient_id, name) if name else None

        seg_name = find_seg_file(patient_dir, patient_id)
        if seg_name is None:
            problems.append(f"{patient_id}: tumour mask missing")
        row["seg"] = os.path.join(patient_id, seg_name) if seg_name else None

        rows.append(row)

    if problems:
        print("ERROR: some patients have missing files:")
        for p in problems:
            print("  -", p)
        sys.exit("Fix the data (or check --data-dir) and run again.")

    return pd.DataFrame(rows)


def split_patients(patients, val_fraction, test_fraction, seed):
    """Two stratified splits: first take out the test set, then the validation set."""
    n_total = len(patients)
    n_test = round(test_fraction * n_total)
    n_val = round(val_fraction * n_total)

    train_val, test = train_test_split(patients,
                                       test_size=n_test,
                                       stratify=patients["label"],
                                       random_state=seed)

    train, val = train_test_split(train_val,
                                  test_size=n_val,
                                  stratify=train_val["label"],
                                  random_state=seed)

    patients = patients.copy()
    patients["split"] = ""
    patients.loc[train.index, "split"] = "train"
    patients.loc[val.index, "split"] = "val"
    patients.loc[test.index, "split"] = "test"
    return patients


def main():
    parser = argparse.ArgumentParser(description="Create the train/val/test patient split.")
    parser.add_argument("--data-dir", required=True,
                        help="Folder containing name_mapping.csv and the BraTS20_Training_XXX folders")
    parser.add_argument("--output", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits.csv"),
                        help="Where to save the split (default: splits.csv next to this script)")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing splits.csv (normally you should NOT do this)")
    args = parser.parse_args()

    # The split must never change once training has started
    if os.path.exists(args.output) and not args.force:
        sys.exit(f"ERROR: {args.output} already exists. The split should only be made once.\n"
                 "Use --force only if you really want to replace it.")

    print(f"Reading patients from: {args.data_dir}")
    patients = collect_patients(args.data_dir)
    print(f"Found {len(patients)} patients with all 5 files.")

    patients = split_patients(patients, args.val_fraction, args.test_fraction, args.seed)

    columns = ["subject_id", "grade", "label", "split"] + MODALITIES + ["seg"]
    patients[columns].to_csv(args.output, index=False)

    # Summary: patients per split and grade
    summary = pd.crosstab(patients["split"], patients["grade"], margins=True, margins_name="total")
    summary = summary.reindex(["train", "val", "test", "total"])
    print("\nPatients per split:")
    print(summary.to_string())
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
