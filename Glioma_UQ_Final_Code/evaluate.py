"""Evaluate the predictions saved by predict_mc.py.

Reports balanced accuracy, AUC, ECE, Brier score and NLL, applies temperature
scaling, draws reliability diagrams and referral curves (against a random
referral baseline), and adds stratified bootstrap confidence intervals.
No GPU is needed.

Usage: python evaluate.py --split val [--uq-dir uq_cv]
The test split also needs --allow-test.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")           # no display on a compute node
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

VARIANTS = ["whole-brain", "mask-guided"]
CONFIGS = ["softmax", "mc-dropout", "deep-ensemble", "combined"]
CONFIG_COLOURS = {"softmax": "#8c8c8c", "mc-dropout": "#4c72b0",
                  "deep-ensemble": "#55a868", "combined": "#c44e52"}
N_BOOTSTRAP = 1000
N_BINS = 10
EPSILON = 1e-12
DPI = 150


# --- the honesty measures --------------------------------------------------

def entropy_bits(probabilities, axis=-1):
    """Shannon entropy in bits; for two classes the maximum is exactly 1."""
    safe = np.clip(probabilities, EPSILON, 1.0)
    return -(safe * np.log2(safe)).sum(axis=axis)


def calibration_bins(probabilities, labels, n_bins=N_BINS):
    """Expected calibration error and the per-bin table for the reliability diagram.
    Confidence is the probability of the predicted class.
    """
    confidence = probabilities.max(axis=1)
    correct = (probabilities.argmax(axis=1) == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    error, rows = 0.0, []
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > low) & (confidence <= high)
        count = int(in_bin.sum())
        if count == 0:
            continue
        mean_confidence = float(confidence[in_bin].mean())
        accuracy = float(correct[in_bin].mean())
        error += (count / len(labels)) * abs(accuracy - mean_confidence)
        rows.append({"bin_low": float(low), "bin_high": float(high), "count": count,
                     "mean_confidence": mean_confidence, "accuracy": accuracy})
    return float(error), rows


def brier(probabilities, labels):
    """Mean squared error of the HGG probability. 0 is perfect, lower is better."""
    return float(np.mean((probabilities[:, 1] - labels) ** 2))


# --- temperature scaling ---------------------------------------------------


def apply_temperature(probabilities, temperature):
    """Divide the (pseudo-)logits by T and re-normalise."""
    logits = np.log(np.clip(probabilities, EPSILON, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)          # overflow guard only
    exponentiated = np.exp(logits)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def fit_temperature(probabilities, labels, n_grid=601, low=0.05, high=10.0):
    """Temperature that minimises NLL, found by a grid search on a log scale."""
    if len(np.unique(labels)) < 2:
        return 1.0
    grid = np.exp(np.linspace(np.log(low), np.log(high), n_grid))
    losses = [negative_log_likelihood(apply_temperature(probabilities, t), labels)
              for t in grid]
    return float(grid[int(np.argmin(losses))])


def cross_fitted_temperature(probabilities, labels, n_folds=5, seed=0):
    """Temperature scaling where each patient is scaled with a temperature fitted
    on the other folds, so no patient affects their own scaling.
    Returns the scaled probabilities and the per-fold temperatures.
    """
    from sklearn.model_selection import StratifiedKFold

    n_folds = max(2, min(n_folds, int(np.bincount(labels).min())))
    scaled = np.empty_like(probabilities)
    temperatures = []

    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fit_index, apply_index in splitter.split(probabilities, labels):
        temperature = fit_temperature(probabilities[fit_index], labels[fit_index])
        scaled[apply_index] = apply_temperature(probabilities[apply_index], temperature)
        temperatures.append(temperature)

    return scaled, temperatures


def negative_log_likelihood(probabilities, labels):
    chosen = probabilities[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(np.clip(chosen, EPSILON, 1.0))))


def balanced_accuracy(labels, probabilities):
    return float(balanced_accuracy_score(labels, probabilities.argmax(axis=1)))


def auc(labels, probabilities):
    try:
        return float(roc_auc_score(labels, probabilities[:, 1]))
    except ValueError:
        return float("nan")


# --- confidence intervals --------------------------------------------------

def bootstrap_interval(metric, labels, probabilities, rng,
                       n_resamples=N_BOOTSTRAP, alpha=0.05):
    """95% percentile interval from bootstrap resamples stratified by grade."""
    pools = [np.flatnonzero(labels == grade) for grade in np.unique(labels)]
    values = []
    for _ in range(n_resamples):
        drawn = np.concatenate([rng.choice(pool, size=len(pool), replace=True)
                                for pool in pools])
        value = metric(labels[drawn], probabilities[drawn])
        if np.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    low, high = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


# --- referral --------------------------------------------------------------

def referral_curves(uncertainty, probabilities, labels, n_points=17):
    """Accuracy, balanced accuracy and LGG count left after referring the most
    uncertain patients, at retention levels from 100% down to 30%.
    """
    order = np.argsort(uncertainty, kind="stable")      # most certain first
    minority = int(np.bincount(labels).argmin())
    total = len(labels)

    rows = []
    for fraction in np.linspace(1.0, 0.3, n_points):
        keep = max(int(round(fraction * total)), 2)
        chosen = order[:keep]
        kept = labels[chosen]
        predicted = probabilities[chosen].argmax(axis=1)
        rows.append({
            "retained": keep / total,
            "accuracy": float((predicted == kept).mean()),
            "balanced_accuracy": (balanced_accuracy(kept, probabilities[chosen])
                                  if len(np.unique(kept)) == 2 else float("nan")),
            "minority_kept": int((kept == minority).sum()),
        })
    return rows


def random_referral_baseline(probabilities, labels, retained, rng, repeats=200):
    """The same retention levels, referring randomly chosen patients instead."""
    total = len(labels)
    accuracy, balanced = [], []
    for fraction in retained:
        keep = max(int(round(fraction * total)), 2)
        drawn_accuracy, drawn_balanced = [], []
        for _ in range(repeats):
            chosen = rng.choice(total, size=keep, replace=False)
            kept = labels[chosen]
            drawn_accuracy.append(float((probabilities[chosen].argmax(axis=1) == kept).mean()))
            if len(np.unique(kept)) == 2:
                drawn_balanced.append(balanced_accuracy(kept, probabilities[chosen]))
        accuracy.append(float(np.mean(drawn_accuracy)))
        balanced.append(float(np.mean(drawn_balanced)) if drawn_balanced else float("nan"))
    return accuracy, balanced


def entropy_ranks_of_errors(uncertainty, probabilities, labels):
    """Rank of each misclassified patient in the uncertainty ordering
    (rank 1 = most uncertain).
    """
    rank = np.empty(len(labels), dtype=int)
    rank[np.argsort(-uncertainty, kind="stable")] = np.arange(1, len(labels) + 1)
    wrong = np.flatnonzero(probabilities.argmax(axis=1) != labels)
    grades = {0: "LGG", 1: "HGG"}
    return [{"rank": int(rank[i]), "of": len(labels), "true_grade": grades[int(labels[i])],
             "prob_hgg": float(probabilities[i, 1]), "entropy": float(uncertainty[i])}
            for i in sorted(wrong, key=lambda index: rank[index])]


# --- loading ---------------------------------------------------------------

def load_configurations(work_dir, variant, split, uq_dir="uq"):
    """Load probabilities.npz for one variant and rebuild the four
    configurations the same way predict_mc.py does.
    """
    folder = os.path.join(work_dir, uq_dir, f"{variant}_{split}")
    archive = os.path.join(folder, "probabilities.npz")
    if not os.path.isfile(archive):
        return None

    with np.load(archive, allow_pickle=False) as saved:
        deterministic = saved["deterministic"]          # [M, N, 2]
        stochastic = saved["stochastic"]                # [M, T, N, 2]
        labels = saved["labels"].astype(int)            # [N]
        seeds = saved["seeds"].astype(int)              # [M]

    primary = 0
    summary_path = os.path.join(folder, "uq_summary.json")
    if os.path.isfile(summary_path):
        with open(summary_path) as handle:
            wanted = json.load(handle).get("primary_seed")
        if wanted in seeds.tolist():
            primary = seeds.tolist().index(wanted)

    n_models, n_passes = stochastic.shape[0], stochastic.shape[1]
    return {"labels": labels, "seeds": seeds, "primary_seed": int(seeds[primary]),
            "models": int(n_models), "passes": int(n_passes),
            "configurations": {
                "softmax":       deterministic[primary][None],
                "mc-dropout":    stochastic[primary],
                "deep-ensemble": deterministic,
                "combined":      stochastic.reshape(n_models * n_passes, -1, 2)}}


# --- figures ---------------------------------------------------------------

def reliability_figure(results, split, out_dir, suffix=""):
    if not results:
        return
    # Axis follows the data, so a low bin is not cut off.
    plotted = [value
               for entry in results.values()
               for rows in (entry["bins"], entry.get("temperature", {}).get("bins", []))
               for row in rows
               for value in (row["accuracy"], row["mean_confidence"])]
    lower = max(0.0, min([0.45] + plotted) - 0.05)

    fig, axes = plt.subplots(len(VARIANTS), len(CONFIGS), figsize=(13, 6.6),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for row, variant in enumerate(VARIANTS):
        for column, config in enumerate(CONFIGS):
            axis = axes[row, column]
            entry = results.get((variant, config))
            if entry is None:
                axis.set_visible(False)
                continue
            bins = pd.DataFrame(entry["bins"])
            axis.plot([lower, 1.0], [lower, 1.0], color="#999999", linestyle="--", linewidth=1)
            if not bins.empty:
                # Marker size shows how many patients are in the bin.
                axis.plot(bins["mean_confidence"], bins["accuracy"],
                          color=CONFIG_COLOURS[config], linewidth=1.0, alpha=.45,
                          zorder=1)
                axis.scatter(bins["mean_confidence"], bins["accuracy"],
                             s=18 + 4.0 * bins["count"], color=CONFIG_COLOURS[config],
                             edgecolor="white", linewidth=.8, zorder=2)
                for _, bin_row in bins.iterrows():
                    axis.annotate(int(bin_row["count"]),
                                  (bin_row["mean_confidence"], bin_row["accuracy"]),
                                  textcoords="offset points", xytext=(0, -14),
                                  ha="center", fontsize=7, color="#555555")

            # Same curve after temperature scaling.
            scaled = pd.DataFrame(entry.get("temperature", {}).get("bins", []))
            if not scaled.empty:
                axis.plot(scaled["mean_confidence"], scaled["accuracy"],
                          color=CONFIG_COLOURS[config], linewidth=1.0,
                          linestyle=":", marker="x", markersize=5, alpha=.9,
                          zorder=3, label="after scaling")

            title = f"{config}\nECE {entry['ece']:.4f}"
            if "temperature" in entry:
                title += f" -> {entry['temperature']['ece']:.4f} scaled"
            axis.set_title(title, fontsize=9)
            axis.set_xlim(lower, 1.02)
            axis.set_ylim(lower - 0.03, 1.03)
            axis.grid(alpha=.25)
            if column == 0:
                axis.set_ylabel(f"{variant}\n\nobserved accuracy", fontsize=9)
            if row == len(VARIANTS) - 1:
                axis.set_xlabel("stated confidence")
    fig.suptitle("Reliability diagrams - the dashed line is perfect honesty, the dotted "
                 "crosses are after temperature scaling  "
                 f"({split} split; marker size and label are the patient count)", fontsize=11)
    path = os.path.join(out_dir, f"fig8_reliability{suffix}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  wrote {path}")


def referral_figure(results, split, out_dir, suffix=""):
    variants = [variant for variant in VARIANTS
                if any((variant, config) in results for config in CONFIGS)]
    if not variants:
        return
    panels = (("accuracy", "accuracy (risk-coverage)"),
              ("balanced_accuracy", "balanced accuracy"))
    fig, axes = plt.subplots(len(variants), 2, figsize=(13, 4.8 * len(variants)),
                             squeeze=False)
    for row, variant in enumerate(variants):
        for column, (key, title) in enumerate(panels):
            axis = axes[row][column]
            baseline_drawn = False
            for config in CONFIGS:
                entry = results.get((variant, config))
                if entry is None or not entry["referral"]:
                    continue
                curve = pd.DataFrame(entry["referral"])
                axis.plot(curve["retained"], curve[key], marker="o", markersize=3.5,
                          color=CONFIG_COLOURS[config], label=config)
                if not baseline_drawn:
                    axis.plot(curve["retained"], entry["baseline"][key],
                              color="#333333", linestyle=":", linewidth=1.4,
                              label="random referral")
                    # Beyond here only one or two LGG patients are left.
                    thin = curve[curve["minority_kept"] <= 2]
                    if key == "balanced_accuracy" and not thin.empty:
                        axis.axvspan(curve["retained"].min(), thin["retained"].max(),
                                     color="#c44e52", alpha=.08)
                        axis.annotate("2 or fewer LGG kept:\nnot measurable here",
                                      xy=(thin["retained"].max(), 0.02),
                                      xycoords=("data", "axes fraction"),
                                      fontsize=7.5, color="#c44e52", ha="right")
                    baseline_drawn = True
            axis.set_title(f"{variant} - {title}", fontsize=10)
            axis.set_xlabel("fraction of patients kept (the rest referred)")
            axis.invert_xaxis()
            axis.grid(alpha=.25)
            if column == 0:
                axis.set_ylabel("score on the kept patients")
                axis.legend(fontsize=8, loc="lower left")
    fig.suptitle("Referral curves - does sending the most uncertain cases to a human help?  "
                 f"({split} split; referral changes the class balance, so both measures are shown)",
                 fontsize=11)
    path = os.path.join(out_dir, f"fig9_referral{suffix}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  wrote {path}")


# --- main ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Calibration, referral and confidence intervals from predict_mc.py output.")
    parser.add_argument("--work-dir", default=os.environ.get("WORK_DIR", ""),
                        help="Scratch folder holding uq/ (default: $WORK_DIR)")
    parser.add_argument("--out-dir", default="figures",
                        help="Where the two figures go")
    parser.add_argument("--results-dir", default="results",
                        help="Where evaluation_<split>.csv and .json go")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--uq-dir", default="uq",
                        help="Folder of predict_mc.py outputs under --work-dir. "
                             "Use uq_cv for the pooled cross-validation results.")
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP,
                        help="Resamples per interval (0 skips the intervals)")
    parser.add_argument("--bins", type=int, default=N_BINS,
                        help="Confidence bins for ECE and the reliability diagram")
    parser.add_argument("--temperature-folds", type=int, default=5,
                        help="Folds used to cross-fit the temperature, so no patient "
                             "influences their own scaling. Capped at the minority "
                             "class size.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for the bootstrap and the random referral baseline")
    parser.add_argument("--allow-test", action="store_true",
                        help="Required for --split test. The test set is opened once, "
                             "for the final numbers that go in the dissertation.")
    args = parser.parse_args()

    # Keep a cross-validated run from overwriting the single-split figures.
    suffix = "" if args.uq_dir == "uq" else f"_{args.uq_dir}"
    # The test figures must not land on top of the validation ones of the same name.
    if args.split == "test":
        suffix += "_test"

    if args.split == "test" and not args.allow_test:
        sys.exit("ERROR: --split test needs --allow-test.\n"
                 "The test set is meant to be opened once, at the end. Use --split val "
                 "while you are still making decisions.")

    if not args.work_dir:
        sys.exit("ERROR: no --work-dir given and $WORK_DIR is not set. "
                 "Set WORK_DIR to the folder holding the uq/ outputs.")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"work dir : {args.work_dir}")
    print(f"split    : {args.split}")
    print(f"bootstrap: {args.bootstrap} stratified resamples\n")

    rows, results, meta = [], {}, {}
    for variant in VARIANTS:
        loaded = load_configurations(args.work_dir, variant, args.split, args.uq_dir)
        if loaded is None:
            print(f"skipped {variant}: no {args.uq_dir}/{variant}_{args.split}/probabilities.npz")
            continue

        labels = loaded["labels"]
        # For the test split, also fit T on validation and apply it to test (Guo et al., 2017).
        validation = (load_configurations(args.work_dir, variant, "val", args.uq_dir)
                      if args.split == "test" else None)
        if args.split == "test" and validation is None:
            print(f"  note: no {args.uq_dir}/{variant}_val/probabilities.npz, so no "
                  f"validation-fitted temperature for {variant}")
        meta[variant] = {"patients": int(len(labels)),
                         "hgg": int((labels == 1).sum()), "lgg": int((labels == 0).sum()),
                         "models": loaded["models"], "passes": loaded["passes"],
                         "primary_seed": loaded["primary_seed"],
                         "seeds": loaded["seeds"].tolist()}
        print(f"{variant}: {len(labels)} patients "
              f"({meta[variant]['hgg']} HGG / {meta[variant]['lgg']} LGG), "
              f"{loaded['models']} models x {loaded['passes']} passes")

        for config, samples in loaded["configurations"].items():
            mean_probabilities = samples.mean(axis=0)               # [N, 2]
            uncertainty = entropy_bits(mean_probabilities)          # predictive entropy

            error, bins = calibration_bins(mean_probabilities, labels, args.bins)
            row = {"variant": variant, "config": config,
                   "samples_per_patient": int(samples.shape[0]),
                   "balanced_accuracy": balanced_accuracy(labels, mean_probabilities),
                   "auc": auc(labels, mean_probabilities),
                   "ece": error,
                   "brier": brier(mean_probabilities, labels),
                   "nll": negative_log_likelihood(mean_probabilities, labels),
                   "mean_predictive_entropy": float(uncertainty.mean())}

            # Temperature scaling, cross-fitted.
            scaled, temperatures = cross_fitted_temperature(
                mean_probabilities, labels, args.temperature_folds, args.seed)
            scaled_error, scaled_bins = calibration_bins(scaled, labels, args.bins)
            row["temperature_in_sample"] = fit_temperature(mean_probabilities, labels)
            row["temperature_cross_fitted_mean"] = float(np.mean(temperatures))
            row["temperature_cross_fitted_sd"] = (float(np.std(temperatures, ddof=1))
                                                  if len(temperatures) > 1 else 0.0)
            row["ece_temperature_scaled"] = scaled_error
            row["brier_temperature_scaled"] = brier(scaled, labels)
            row["nll_temperature_scaled"] = negative_log_likelihood(scaled, labels)
            # Accuracy and AUC after scaling (should barely change).
            row["balanced_accuracy_temperature_scaled"] = balanced_accuracy(labels, scaled)
            row["auc_temperature_scaled"] = auc(labels, scaled)
            if validation is not None:
                val_mean = validation["configurations"][config].mean(axis=0)
                val_temperature = fit_temperature(val_mean, validation["labels"])
                from_val = apply_temperature(mean_probabilities, val_temperature)
                row["temperature_from_val"] = val_temperature
                row["ece_temperature_from_val"] = calibration_bins(from_val, labels,
                                                                   args.bins)[0]
                row["brier_temperature_from_val"] = brier(from_val, labels)
                row["nll_temperature_from_val"] = negative_log_likelihood(from_val, labels)

            if args.bootstrap:
                intervals = (("balanced_accuracy", balanced_accuracy),
                             ("auc", auc),
                             ("ece", lambda y, p: calibration_bins(p, y, args.bins)[0]),
                             ("brier", lambda y, p: brier(p, y)))
                for name, metric in intervals:
                    low, high = bootstrap_interval(metric, labels, mean_probabilities,
                                                   rng, args.bootstrap)
                    row[f"{name}_lo"], row[f"{name}_hi"] = low, high

                low, high = bootstrap_interval(
                    lambda y, p: calibration_bins(p, y, args.bins)[0],
                    labels, scaled, rng, args.bootstrap)
                row["ece_temperature_scaled_lo"] = low
                row["ece_temperature_scaled_hi"] = high

            curve = referral_curves(uncertainty, mean_probabilities, labels)
            retained = np.array([point["retained"] for point in curve])
            base_accuracy, base_balanced = random_referral_baseline(
                mean_probabilities, labels, retained, rng)
            for level in (0.9, 0.8, 0.7):
                nearest = int(np.argmin(np.abs(retained - level)))
                tag = f"at_{int(level * 100)}pct_kept"
                row[f"accuracy_{tag}"] = curve[nearest]["accuracy"]
                row[f"bal_acc_{tag}"] = curve[nearest]["balanced_accuracy"]
                row[f"lgg_kept_{tag}"] = curve[nearest]["minority_kept"]

            # Uncertainty rank of each wrong prediction (1 = most uncertain).
            mistakes = entropy_ranks_of_errors(uncertainty, mean_probabilities, labels)
            row["errors"] = len(mistakes)
            row["error_entropy_ranks"] = "; ".join(
                f"{m['true_grade']} {m['rank']}/{m['of']}" for m in mistakes)

            rows.append(row)
            results[(variant, config)] = {
                "ece": error, "bins": bins, "referral": curve, "errors": mistakes,
                "baseline": {"accuracy": base_accuracy, "balanced_accuracy": base_balanced},
                "temperature": {"in_sample": row["temperature_in_sample"],
                                "cross_fitted": temperatures,
                                "ece": scaled_error, "bins": scaled_bins}}

            print(f"  {config:14s} bal.acc {row['balanced_accuracy']:.4f}  "
                  f"AUC {row['auc']:.4f}  ECE {row['ece']:.4f}  "
                  f"Brier {row['brier']:.4f}  NLL {row['nll']:.4f}")
            print(f"  {'':14s} after temperature scaling: T {row['temperature_cross_fitted_mean']:.3f}"
                  f" +/- {row['temperature_cross_fitted_sd']:.3f}  "
                  f"ECE {scaled_error:.4f} ({scaled_error - error:+.4f})  "
                  f"Brier {row['brier_temperature_scaled']:.4f}  "
                  f"NLL {row['nll_temperature_scaled']:.4f}")
            if "temperature_from_val" in row:
                print(f"  {'':14s} T fitted on val instead: T {row['temperature_from_val']:.3f}  "
                      f"ECE {row['ece_temperature_from_val']:.4f}  "
                      f"Brier {row['brier_temperature_from_val']:.4f}  "
                      f"NLL {row['nll_temperature_from_val']:.4f}")
        print()

    if not rows:
        sys.exit("ERROR: nothing to evaluate. Run predict_mc.py first.")

    table = pd.DataFrame(rows)
    csv_path = os.path.join(args.results_dir, f"evaluation_{args.split}.csv" if args.uq_dir == "uq" else f"evaluation_{args.uq_dir}_{args.split}.csv")
    table.to_csv(csv_path, index=False)
    print(f"  wrote {csv_path}")

    json_path = os.path.join(args.results_dir, f"evaluation_{args.split}.json" if args.uq_dir == "uq" else f"evaluation_{args.uq_dir}_{args.split}.json")
    with open(json_path, "w") as handle:
        json.dump({"split": args.split, "bootstrap_resamples": args.bootstrap,
                   "confidence_bins": args.bins, "seed": args.seed,
                   "temperature_folds": args.temperature_folds,
                   "temperature_note": "T cross-fitted within the split; log(p) used as "
                                       "the logit vector because predict_mc.py saved "
                                       "probabilities, not logits",
                   "splits": meta, "metrics": rows,
                   "curves": {f"{variant}|{config}": entry
                              for (variant, config), entry in results.items()}},
                  handle, indent=2)
    print(f"  wrote {json_path}")

    reliability_figure(results, args.split, args.out_dir, suffix)
    referral_figure(results, args.split, args.out_dir, suffix)

    print("\nLower is better for ECE, Brier and NLL. Higher is better for the rest.")
    print("A referral curve only means something where it rises above the dotted "
          "random-referral line.")
    print("Temperature scaling is the comparison that matters for calibration: if one "
          "cross-fitted scalar reaches the same ECE as MC Dropout or a deep ensemble, "
          "say so - it is a cheaper method and the honest result.")
    print("error_entropy_ranks gives each mistake's place in the uncertainty ordering: "
          "rank 1 is the most uncertain patient, so a high rank number is a confidently "
          "wrong prediction that referral cannot catch.")


if __name__ == "__main__":
    main()
