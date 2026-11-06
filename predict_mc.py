"""Predictions with uncertainty from the trained graders.

Four configurations are computed in one pass:
    softmax        1 model,  1 pass
    mc-dropout     1 model,  T passes with dropout on
    deep-ensemble  M models, 1 pass each
    combined       M models, T passes each

For each patient: predictive entropy (total), aleatoric (mean entropy of the
samples) and epistemic (the difference), in bits. T = 30, dropout 0.25.

Usage: python predict_mc.py --cache-dir CACHE --run-dir RUNS/grader_whole-brain \
                            --out-dir OUT --split val
The test split also needs --allow-test.
"""
import argparse
import glob
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
from monai.data import DataLoader
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score)

from train_grader import CachedGrading, build_model, load_split

DROPOUT_LAYERS = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)
EPSILON = 1e-12


def entropy_bits(probabilities, axis=-1):
    """Shannon entropy in bits. For two classes the maximum is exactly 1."""
    safe = np.clip(probabilities, EPSILON, 1.0)
    return -(safe * np.log2(safe)).sum(axis=axis)


def enable_dropout(model):
    """Switch only the dropout layers to training mode (Monte Carlo dropout).
    Returns how many dropout layers were found.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, DROPOUT_LAYERS):
            module.train()
            count += 1
    return count


def seed_of(checkpoint_path):
    """The seed number in .../seed3/best.pt"""
    folder = os.path.basename(os.path.dirname(checkpoint_path))
    return int("".join(character for character in folder if character.isdigit()) or 0)


def find_checkpoints(run_dir, wanted_seeds):
    """The seed<N>/best.pt files for the requested seeds, sorted by seed.
    Stops if any requested seed is missing, so both variants use the same ensemble.
    """
    found = {seed_of(path): path for path in glob.glob(os.path.join(run_dir, "seed*", "best.pt"))}
    if not found:
        sys.exit(f"ERROR: no seed*/best.pt found in {run_dir}\n"
                 "Train the graders first: train_grader.py with --seed 1 to 5.")

    missing = [seed for seed in wanted_seeds if seed not in found]
    if missing:
        sys.exit(f"ERROR: no trained model for seed(s) {missing} in {run_dir}. "
                 f"Available: {sorted(found)}.\n"
                 "Either finish training them or pass --seeds with the ones you have. "
                 "Both variants must use the same list, or the ablation is not a fair test.")

    ignored = sorted(set(found) - set(wanted_seeds))
    if ignored:
        print(f"  (ignoring seed(s) {ignored} - not in --seeds)")
    return [found[seed] for seed in sorted(wanted_seeds)]


def predict_one_model(checkpoint_path, loader, device, passes, use_amp, mc_seed):
    """Deterministic [N, 2] and stochastic [T, N, 2] probabilities for one model."""
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    settings = state.get("args", {})
    model = build_model(in_channels=5 if settings.get("variant") == "mask-guided" else 4,
                        dropout=settings.get("dropout", 0.25)).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    if enable_dropout(model) == 0:
        sys.exit(f"ERROR: {checkpoint_path} has no dropout layers, so Monte Carlo Dropout "
                 "cannot be run. It was trained with --dropout 0.")
    model.eval()                      # back off again; switched on per pass below

    torch.manual_seed(mc_seed)        # reproducible dropout masks
    deterministic, stochastic = [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)

            model.eval()              # dropout OFF - the ordinary prediction
            with torch.autocast("cuda", enabled=use_amp):
                logits = model(images)
            deterministic.append(torch.softmax(logits.float(), dim=1).cpu().numpy())

            enable_dropout(model)     # dropout ON - T noisy predictions
            batch_passes = []
            for _ in range(passes):
                with torch.autocast("cuda", enabled=use_amp):
                    logits = model(images)
                batch_passes.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            stochastic.append(np.stack(batch_passes))          # [T, batch, 2]
            model.eval()

    return np.concatenate(deterministic), np.concatenate(stochastic, axis=1)


def score(true_labels, probabilities):
    """Classification metrics for one configuration's mean probabilities."""
    predicted = probabilities.argmax(axis=1)
    matrix = confusion_matrix(true_labels, predicted, labels=[0, 1])
    (true_lgg, false_hgg), (false_lgg, true_hgg) = matrix
    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(true_labels, predicted)),
        "accuracy": float(np.mean(true_labels == predicted)),
        "f1_hgg": float(f1_score(true_labels, predicted, pos_label=1, zero_division=0)),
        "sensitivity_hgg": float(true_hgg / (true_hgg + false_lgg)) if (true_hgg + false_lgg) else float("nan"),
        "specificity_lgg": float(true_lgg / (true_lgg + false_hgg)) if (true_lgg + false_hgg) else float("nan"),
        "confusion": matrix.tolist(),
    }
    try:
        metrics["auc"] = float(roc_auc_score(true_labels, probabilities[:, 1]))
    except ValueError:
        metrics["auc"] = float("nan")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Predict with uncertainty from the trained graders.")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--run-dir", required=True,
                        help="Folder holding seed1/best.pt ... seed5/best.pt")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--passes", type=int, default=30,
                        help="T, the number of Monte Carlo Dropout passes")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                        help="Which trained seeds form the ensemble. Must be the same list for "
                             "every variant, otherwise the ablation compares ensemble sizes "
                             "instead of inputs. The seed 42 baseline is excluded by default.")
    parser.add_argument("--primary-seed", type=int, default=None,
                        help="Which single model the softmax and mc-dropout rows use. "
                             "Default: the lowest seed present.")
    parser.add_argument("--mask-dir", default=None,
                        help="Override the predicted-mask folder. Normally read from the checkpoint.")
    parser.add_argument("--batch-size", type=int, default=4)
    folds.add_fold_arguments(parser, os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--allow-test", action="store_true",
                        help="Required to run on the test split. The test set is opened once, "
                             "for the final numbers that go in the dissertation.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="4 patients, 3 passes, CPU allowed, written to a smoke_test subfolder.")
    args = parser.parse_args()

    if args.smoke_test:
        args.limit = args.limit or 4
        args.passes = min(args.passes, 3)
        args.batch_size = 1
        args.workers = 0
        args.out_dir = os.path.join(args.out_dir, "smoke_test")
        print("SMOKE TEST: tiny run to check the code only. Do not report these numbers.\n")

    if args.split == "test" and not args.allow_test:
        sys.exit("ERROR: --split test needs --allow-test.\n"
                 "The test set is meant to be opened once, at the end. Use --split val while "
                 "you are still making decisions.")

    os.makedirs(args.out_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"torch {torch.__version__} (CUDA build {torch.version.cuda}) | MONAI {monai.__version__}")
        print(f"GPU: {torch.cuda.get_device_properties(0).name}")
    elif args.smoke_test:
        device = torch.device("cpu")
        print("No GPU visible - running the smoke test on CPU.")
    else:
        sys.exit("ERROR: no GPU visible. Submit with sbatch, or add --smoke-test.")
    use_amp = not args.no_amp and device.type == "cuda"

    # --- Models -------------------------------------------------------------
    checkpoints = find_checkpoints(args.run_dir, args.seeds)
    first = torch.load(checkpoints[0], map_location="cpu", weights_only=False).get("args", {})
    variant = first.get("variant", "whole-brain")
    mask_dir = args.mask_dir or (first.get("mask_dir") if variant == "mask-guided" else None)
    seeds = [seed_of(path) for path in checkpoints]
    print(f"{len(checkpoints)} models ({variant}), seeds {seeds} | split {args.split} "
          f"| T = {args.passes} passes")
    if len(checkpoints) < 2:
        print("WARNING: only one model found, so the deep-ensemble and combined rows are not "
              "real ensembles. Train the other seeds before quoting them.")

    # --- Data (with --fold: the patients held out at that fold) ---------------
    split = "val" if args.fold is not None else args.split
    records = load_split(args.cache_dir, split, args.limit, args.fold, args.folds_file)
    files = [filename for filename, _ in records]
    true_labels = np.array([label for _, label in records])
    loader = DataLoader(
        CachedGrading(args.cache_dir, records, transform=None, crop=None, mask_dir=mask_dir),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=device.type == "cuda")
    print(f"{len(records)} patients ({int((true_labels == 1).sum())} HGG / "
          f"{int((true_labels == 0).sum())} LGG)")

    # --- Predict ------------------------------------------------------------
    started = time.time()
    all_deterministic, all_stochastic = [], []
    for seed, path in zip(seeds, checkpoints):
        model_start = time.time()
        deterministic, stochastic = predict_one_model(
            path, loader, device, args.passes, use_amp, mc_seed=1000 + seed)
        all_deterministic.append(deterministic)
        all_stochastic.append(stochastic)
        print(f"  seed {seed}: {1 + args.passes} passes over {len(records)} patients "
              f"in {time.time() - model_start:.1f}s", flush=True)

    deterministic = np.stack(all_deterministic)        # [M, N, 2]
    stochastic = np.stack(all_stochastic)              # [M, T, N, 2]
    n_models, n_passes = stochastic.shape[0], stochastic.shape[1]

    primary = seeds.index(args.primary_seed) if args.primary_seed in seeds else 0
    if args.primary_seed is not None and args.primary_seed not in seeds:
        print(f"WARNING: seed {args.primary_seed} not found, using seed {seeds[primary]}.")

    # --- The four configurations, each a stack of probability vectors -------
    configurations = {
        "softmax":       deterministic[primary][None],                       # [1, N, 2]
        "mc-dropout":    stochastic[primary],                                # [T, N, 2]
        "deep-ensemble": deterministic,                                      # [M, N, 2]
        "combined":      stochastic.reshape(n_models * n_passes, -1, 2),     # [M*T, N, 2]
    }

    rows, summary = [], {}
    for name, samples in configurations.items():
        mean_probabilities = samples.mean(axis=0)                      # [N, 2]
        predictive = entropy_bits(mean_probabilities)                  # total
        aleatoric = entropy_bits(samples).mean(axis=0)                 # data noise
        epistemic = predictive - aleatoric                             # model ignorance
        predicted = mean_probabilities.argmax(axis=1)

        metrics = score(true_labels, mean_probabilities)
        metrics.update({"samples_per_patient": int(samples.shape[0]),
                        "mean_predictive_entropy": float(predictive.mean()),
                        "mean_aleatoric": float(aleatoric.mean()),
                        "mean_epistemic": float(epistemic.mean())})
        summary[name] = metrics

        for index, filename in enumerate(files):
            rows.append({"file": filename,
                         "split": args.split,
                         "config": name,
                         "true_label": int(true_labels[index]),
                         "true_grade": "HGG" if true_labels[index] == 1 else "LGG",
                         "predicted_label": int(predicted[index]),
                         "correct": bool(predicted[index] == true_labels[index]),
                         "prob_hgg": float(mean_probabilities[index, 1]),
                         "predictive_entropy": float(predictive[index]),
                         "aleatoric": float(aleatoric[index]),
                         "epistemic": float(epistemic[index]),
                         "std_prob_hgg": float(samples[:, index, 1].std())})

        print(f"{name:14s} bal.acc {metrics['balanced_accuracy']:.4f}  "
              f"AUC {metrics['auc']:.4f}  "
              f"entropy {metrics['mean_predictive_entropy']:.4f} "
              f"(alea {metrics['mean_aleatoric']:.4f} epis {metrics['mean_epistemic']:.4f})")

    # --- Save ---------------------------------------------------------------
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.out_dir, "uncertainty.csv"), index=False)

    # Raw probabilities so evaluate.py can bootstrap without re-running the GPU.
    np.savez_compressed(os.path.join(args.out_dir, "probabilities.npz"),
                        deterministic=deterministic, stochastic=stochastic,
                        labels=true_labels, files=np.array(files), seeds=np.array(seeds))

    with open(os.path.join(args.out_dir, "uq_summary.json"), "w") as f:
        json.dump({"finished": datetime.now().isoformat(timespec="seconds"),
                   "run_dir": args.run_dir,
                   "fold": args.fold,
                   "variant": variant,
                   "split": args.split,
                   "patients": len(records),
                   "models": n_models,
                   "seeds": seeds,
                   "passes": n_passes,
                   "primary_seed": seeds[primary],
                   "entropy_units": "bits (max 1.0 for two classes)",
                   "configurations": summary,
                   "torch_version": torch.__version__}, f, indent=2)

    print(f"\nDone in {time.time() - started:.1f}s. Wrote uncertainty.csv, probabilities.npz "
          f"and uq_summary.json to {args.out_dir}")


if __name__ == "__main__":
    main()
