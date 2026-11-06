"""Join the five per-fold outputs of predict_mc.py into one file, so every
non-test patient has an out-of-fold prediction (314 patients).

Usage: python pool_folds.py --uq-dir uq_cv
"""
import argparse
import json
import os
import sys

import numpy as np

VARIANTS = ["whole-brain", "mask-guided"]


def pool_variant(work_dir, uq_dir, variant, n_folds):
    """Concatenate the per-fold runs for one variant along the patient axis."""
    deterministic, stochastic, labels, files, folds_used = [], [], [], [], []
    shapes = set()

    for fold in range(n_folds):
        archive = os.path.join(work_dir, uq_dir, f"{variant}_fold{fold}", "probabilities.npz")
        if not os.path.isfile(archive):
            return None, f"missing {archive}"

        with np.load(archive, allow_pickle=False) as saved:
            deterministic.append(saved["deterministic"])          # [M, n_k, 2]
            stochastic.append(saved["stochastic"])                # [M, T, n_k, 2]
            labels.append(saved["labels"])
            files.append(saved["files"])
            seeds = saved["seeds"]
        # Models and passes must match, or the folds are not comparable.
        shapes.add((deterministic[-1].shape[0], stochastic[-1].shape[1], tuple(seeds)))
        folds_used += [fold] * len(labels[-1])

    if len(shapes) != 1:
        return None, f"folds disagree on (models, passes, seeds): {sorted(shapes)}"

    pooled = {
        "deterministic": np.concatenate(deterministic, axis=1),   # [M, N, 2]
        "stochastic": np.concatenate(stochastic, axis=2),         # [M, T, N, 2]
        "labels": np.concatenate(labels),
        "files": np.concatenate(files),
        "seeds": seeds,
        "fold_of_patient": np.array(folds_used),
    }

    if len(set(pooled["files"].tolist())) != len(pooled["files"]):
        return None, "a patient appears in more than one fold"

    return pooled, None


def main():
    parser = argparse.ArgumentParser(
        description="Pool per-fold predict_mc.py runs into one out-of-fold set.")
    parser.add_argument("--work-dir", default=os.environ.get("WORK_DIR", ""))
    parser.add_argument("--uq-dir", default="uq_cv",
                        help="Folder holding <variant>_fold<k> runs; the pooled result "
                             "is written to <variant>_val inside it")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing pooled file")
    args = parser.parse_args()

    if not args.work_dir:
        sys.exit("ERROR: no --work-dir and $WORK_DIR is not set. "
                 "Set WORK_DIR to the folder holding the uq/ outputs.")

    wrote = 0
    for variant in VARIANTS:
        pooled, problem = pool_variant(args.work_dir, args.uq_dir, variant, args.folds)
        if pooled is None:
            print(f"skipped {variant}: {problem}")
            continue

        out_dir = os.path.join(args.work_dir, args.uq_dir, f"{variant}_val")
        archive = os.path.join(out_dir, "probabilities.npz")
        if os.path.exists(archive) and not args.force:
            print(f"skipped {variant}: {archive} exists (use --force)")
            continue
        os.makedirs(out_dir, exist_ok=True)

        labels = pooled["labels"]
        hgg = int((labels == 1).sum())
        print(f"{variant}: {len(labels)} patients ({hgg} HGG / {len(labels) - hgg} LGG) "
              f"from {args.folds} folds | "
              f"{pooled['deterministic'].shape[0]} models x {pooled['stochastic'].shape[1]} passes")

        np.savez_compressed(archive, **pooled)
        with open(os.path.join(out_dir, "uq_summary.json"), "w") as handle:
            json.dump({"pooled_from_folds": args.folds,
                       "variant": variant,
                       "split": "val (out-of-fold)",
                       "patients": int(len(labels)),
                       "hgg": hgg, "lgg": int(len(labels) - hgg),
                       "models": int(pooled["deterministic"].shape[0]),
                       "passes": int(pooled["stochastic"].shape[1]),
                       "seeds": pooled["seeds"].tolist(),
                       "primary_seed": int(pooled["seeds"][0]),
                       "entropy_units": "bits (max 1.0 for two classes)"}, handle, indent=2)
        print(f"  wrote {archive}")
        wrote += 1

    if not wrote:
        sys.exit("ERROR: nothing pooled. Run predict_mc.py --fold k for each fold first.")

    print(f"\nEvaluate the pooled predictions with:")
    print(f"  python evaluate.py --split val --uq-dir {args.uq_dir}")


if __name__ == "__main__":
    main()
