"""Draw the figures used in the paper (PDF and 300 dpi PNG).

Most figures only need the files in results/. The entropy figures also need
the per-patient probabilities in $WORK_DIR and are skipped if those are missing.

Usage: python make_paper_figures.py [--out-dir figures/paper]
"""
import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")           # no display on a compute node
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

VARIANTS = ["whole-brain", "mask-guided"]
VARIANT_NAMES = {"whole-brain": "Whole-brain", "mask-guided": "Mask-guided"}
CONFIGS = ["softmax", "mc-dropout", "deep-ensemble", "combined"]
CONFIG_NAMES = {"softmax": "Softmax", "mc-dropout": "MC dropout",
                "deep-ensemble": "Deep ensemble", "combined": "Combined"}
# Two-line tick labels so they fit under a narrow panel.
CONFIG_TICKS = {"softmax": "Softmax", "mc-dropout": "MC\ndropout",
                "deep-ensemble": "Deep\nensemble", "combined": "Combined"}

# Colour helps on screen; marker and line style carry the meaning in print.
CONFIG_STYLE = {
    "softmax":       dict(color="#555555", marker="s", linestyle="--"),
    "mc-dropout":    dict(color="#2b5d9b", marker="^", linestyle="-."),
    "deep-ensemble": dict(color="#1b7f4b", marker="o", linestyle="-"),
    "combined":      dict(color="#b03a3a", marker="D", linestyle=(0, (1, 1))),
}
BASELINE_STYLE = dict(color="black", linestyle=(0, (6, 2)), linewidth=1.0)

FULL_WIDTH = 6.5                # inches; scaled to \textwidth in LaTeX
EPSILON = 1e-12

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 8.5,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8.5,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.1,
    "lines.markersize": 3.5,
})


def save(fig, out_dir, name):
    for extension, extra in (("pdf", {}), ("png", {"dpi": 300})):
        path = os.path.join(out_dir, f"{name}.{extension}")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02, **extra)
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def skip(name, reason):
    print(f"  skipped {name}: {reason}")


def load_curves(path):
    if not os.path.isfile(path):
        return None
    with open(path) as handle:
        return json.load(handle)["curves"]


def entropy_bits(probabilities, axis=-1):
    safe = np.clip(probabilities, EPSILON, 1.0)
    return -(safe * np.log2(safe)).sum(axis=axis)


# --- Figure 1: the pipeline --------------------------------------------------

def pipeline(out_dir):
    fig, axis = plt.subplots(figsize=(FULL_WIDTH, 2.5))
    axis.set_xlim(0, 100)
    axis.set_ylim(0, 37)
    axis.axis("off")

    def box(x, y, width, height, heading, body, fill="white"):
        axis.add_patch(FancyBboxPatch((x, y), width, height,
                                      boxstyle="round,pad=0.25,rounding_size=1.2",
                                      facecolor=fill, edgecolor="black", linewidth=0.7))
        axis.text(x + width / 2, y + height - 1.8, heading, ha="center", va="top",
                  fontweight="bold", fontsize=7.8)
        axis.text(x + width / 2, y + height - 6.0, body, ha="center", va="top",
                  fontsize=6.8, linespacing=1.3)

    def arrow(start, end, label=None, label_offset=(0, 1.2), dashed=False):
        axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9,
                                       linewidth=0.8, color="black",
                                       linestyle=(0, (3, 2)) if dashed else "-"))
        if label:
            axis.text((start[0] + end[0]) / 2 + label_offset[0],
                      (start[1] + end[1]) / 2 + label_offset[1], label,
                      ha="center", va="bottom", fontsize=6.5, style="italic")

    grey = "#ececec"
    box(0, 9, 14, 22, "Input", "Four MRI\nsequences:\nT1, T1ce,\nT2, FLAIR\n4×128×128×96")
    box(19, 21, 18.5, 14, "Stage 1", "3D U-Net\nsegmentation of\nthe whole tumour", fill=grey)
    box(19, 1, 18.5, 14, "Stage 2", "3D CNN grader:\nwhole-brain (4 ch) or\nmask-guided (5 ch)",
        fill=grey)
    box(42.5, 3, 21, 30, "Stage 3: uncertainty",
        "Softmax (1 pass)\nMC dropout (T = 30)\nDeep ensemble (M = 5)\nCombined (5 × 30)\n"
        "\nTemperature scaling\nas a baseline", fill=grey)
    box(68, 3, 15, 30, "Per patient",
        "Grade\n(HGG / LGG)\n\nProbability\n\nEntropy, split\ninto aleatoric\nand epistemic")
    box(87.5, 8, 12.5, 20, "Decision", "Accept,\nor refer to\na radiologist")

    arrow((14.3, 26), (18.7, 28))
    arrow((14.3, 14), (18.7, 8))
    arrow((28.25, 20.7), (28.25, 15.3), "predicted mask", label_offset=(7.2, -1.3), dashed=True)
    arrow((37.8, 8), (42.2, 12))
    arrow((63.8, 18), (67.7, 18))
    arrow((83.3, 18), (87.2, 18))
    save(fig, out_dir, "fig1_pipeline")


# --- Figure 2: confusion matrices -------------------------------------------------

def load_errors(path):
    """Misclassified HGG and LGG counts per variant and configuration."""
    if not os.path.isfile(path):
        return None
    import csv
    errors = {}
    with open(path) as handle:
        for row in csv.DictReader(handle):
            found = re.findall(r"(HGG|LGG) \d+/(\d+)", row["error_entropy_ranks"])
            errors[(row["variant"], row["config"])] = (
                sum(1 for grade, _ in found if grade == "HGG"),
                sum(1 for grade, _ in found if grade == "LGG"))
    return errors


def confusion(cv_errors, test_errors, out_dir, name):
    """One 2x2 matrix per configuration: cross-validation (314) and test (55)."""
    if cv_errors is None or test_errors is None:
        return skip(name, "no evaluation CSV")

    blocks = [("Cross-validation", cv_errors, (249, 65)),
              ("Test set", test_errors, (44, 11))]
    fig, axes = plt.subplots(4, len(CONFIGS), figsize=(FULL_WIDTH, 5.8))
    for block, (label, errors, (n_hgg, n_lgg)) in enumerate(blocks):
        for offset, variant in enumerate(VARIANTS):
            row = 2 * block + offset
            for column, config in enumerate(CONFIGS):
                axis = axes[row, column]
                wrong_hgg, wrong_lgg = errors[(variant, config)]
                # Rows are the true grade, columns the predicted grade.
                counts = np.array([[n_hgg - wrong_hgg, wrong_hgg],
                                   [wrong_lgg, n_lgg - wrong_lgg]])
                share = counts / counts.sum(axis=1, keepdims=True)
                axis.imshow(share, cmap="Greys", vmin=0, vmax=1.25)
                for i in range(2):
                    for j in range(2):
                        axis.text(j, i, f"{counts[i, j]}\n{100 * share[i, j]:.0f}%",
                                  ha="center", va="center", fontsize=7,
                                  color="white" if share[i, j] > 0.6 else "black")
                axis.set_xticks([0, 1])
                axis.set_yticks([0, 1])
                axis.set_xticklabels(["HGG", "LGG"] if row == 3 else [])
                axis.set_yticklabels(["HGG", "LGG"] if column == 0 else [])
                axis.tick_params(length=0)
                if row == 0:
                    axis.set_title(CONFIG_NAMES[config])
                if column == 0:
                    axis.set_ylabel(f"{label}\n{VARIANT_NAMES[variant]}\ntrue grade",
                                    fontsize=7.5)
                if row == 3:
                    axis.set_xlabel("predicted grade")
    fig.tight_layout(h_pad=0.8, w_pad=0.4)
    save(fig, out_dir, name)


# --- Figure 3: reliability diagrams ---------------------------------------------

def reliability(curves, out_dir, name):
    if curves is None:
        return skip(name, "no evaluation JSON")

    fig, axes = plt.subplots(len(VARIANTS), len(CONFIGS), figsize=(FULL_WIDTH, 3.6),
                             sharex=True, sharey=True)
    low = 0.4
    for row, variant in enumerate(VARIANTS):
        for column, config in enumerate(CONFIGS):
            axis = axes[row, column]
            entry = curves.get(f"{variant}|{config}")
            if entry is None:
                axis.set_visible(False)
                continue
            style = CONFIG_STYLE[config]
            axis.plot([low, 1], [low, 1], color="#888888", linewidth=0.7, linestyle=":")

            before = entry["bins"]
            confidence = [b["mean_confidence"] for b in before]
            accuracy = [b["accuracy"] for b in before]
            counts = np.array([b["count"] for b in before])
            axis.plot(confidence, accuracy, color=style["color"], linewidth=0.9)
            axis.scatter(confidence, accuracy, s=6 + 0.35 * counts, color=style["color"],
                         marker=style["marker"], edgecolor="white", linewidth=0.4, zorder=3,
                         label="before scaling")

            after = entry.get("temperature", {}).get("bins", [])
            if after:
                axis.plot([b["mean_confidence"] for b in after],
                          [b["accuracy"] for b in after], color="black", linewidth=0.7,
                          linestyle="--", marker="x", markersize=3, zorder=4,
                          label="after scaling")

            ece_after = entry.get("temperature", {}).get("ece")
            text = f"ECE {entry['ece']:.3f}"
            if ece_after is not None:
                text += f"\n→ {ece_after:.3f}"
            axis.text(0.97, 0.04, text, transform=axis.transAxes, ha="right", va="bottom",
                      fontsize=6.8)
            axis.set_xlim(low, 1.01)
            axis.set_ylim(low, 1.03)
            axis.set_xticks([0.4, 0.6, 0.8, 1.0])
            axis.set_yticks([0.4, 0.6, 0.8, 1.0])
            axis.grid(alpha=0.2, linewidth=0.4)
            if row == 0:
                axis.set_title(CONFIG_NAMES[config])
            if column == 0:
                axis.set_ylabel(f"{VARIANT_NAMES[variant]}\nobserved accuracy")
            if row == len(VARIANTS) - 1:
                axis.set_xlabel("confidence")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.04, 1, 1), w_pad=0.6, h_pad=0.6)
    save(fig, out_dir, name)


# --- Figures 4 and 5: referral curves --------------------------------------------

def referral(curves, out_dir, name):
    if curves is None:
        return skip(name, "no evaluation JSON")

    fig, axes = plt.subplots(1, len(VARIANTS), figsize=(FULL_WIDTH, 2.6), sharey=True)
    for axis, variant in zip(axes, VARIANTS):
        baseline_drawn = False
        for config in CONFIGS:
            entry = curves.get(f"{variant}|{config}")
            if entry is None or not entry["referral"]:
                continue
            # Points with two or fewer LGG patients left are not drawn.
            points = [p for p in entry["referral"] if p["minority_kept"] > 2]
            retained = [p["retained"] for p in points]
            score = [p["balanced_accuracy"] for p in points]
            axis.plot(retained, score, label=CONFIG_NAMES[config], **CONFIG_STYLE[config])
            if not baseline_drawn:
                axis.plot([p["retained"] for p in entry["referral"]],
                          entry["baseline"]["balanced_accuracy"],
                          label="Random referral", **BASELINE_STYLE)
                baseline_drawn = True
        axis.axvline(0.8, color="#999999", linewidth=0.5, zorder=0)
        axis.set_title(VARIANT_NAMES[variant])
        axis.set_xlabel("fraction of patients kept")
        axis.invert_xaxis()
        axis.grid(alpha=0.2, linewidth=0.4)
    axes[0].set_ylabel("balanced accuracy on kept patients")

    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index(n) for n in labels if n != "Random referral"] + \
            [labels.index("Random referral")]
    handles, labels = [handles[i] for i in order], [labels[i] for i in order]
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, out_dir, name)


# --- Figures 6 and 7: per-patient uncertainty ------------------------------------

def load_uncertainty(work_dir, uq_dir, split):
    """Predictive, aleatoric and epistemic entropy per patient and configuration."""
    tables = {}
    for variant in VARIANTS:
        folder = os.path.join(work_dir, uq_dir, f"{variant}_{split}")
        archive = os.path.join(folder, "probabilities.npz")
        if not work_dir or not os.path.isfile(archive):
            continue
        with np.load(archive, allow_pickle=False) as saved:
            deterministic = saved["deterministic"]          # [M, N, 2]
            stochastic = saved["stochastic"]                # [M, T, N, 2]
            labels = saved["labels"].astype(int)
            seeds = saved["seeds"].astype(int).tolist()

        primary = 0
        summary_path = os.path.join(folder, "uq_summary.json")
        if os.path.isfile(summary_path):
            with open(summary_path) as handle:
                wanted = json.load(handle).get("primary_seed")
            if wanted in seeds:
                primary = seeds.index(wanted)

        n_models, n_passes = stochastic.shape[0], stochastic.shape[1]
        samples_by_config = {
            "softmax":       deterministic[primary][None],
            "mc-dropout":    stochastic[primary],
            "deep-ensemble": deterministic,
            "combined":      stochastic.reshape(n_models * n_passes, -1, 2)}

        tables[variant] = {}
        for config, samples in samples_by_config.items():
            mean = samples.mean(axis=0)
            predictive = entropy_bits(mean)
            aleatoric = entropy_bits(samples).mean(axis=0)
            tables[variant][config] = {
                "correct": mean.argmax(axis=1) == labels,
                "predictive": predictive,
                "aleatoric": aleatoric,
                "epistemic": predictive - aleatoric}
    return tables


def entropy_correct_vs_wrong(tables, out_dir, name):
    if not tables:
        return skip(name, "no probabilities.npz (run on the HPC with $WORK_DIR set)")

    fig, axes = plt.subplots(1, len(tables), figsize=(FULL_WIDTH, 2.5), sharey=True,
                             squeeze=False)
    for axis, (variant, table) in zip(axes[0], tables.items()):
        data, positions, wrong = [], [], []
        for index, config in enumerate(CONFIGS):
            values = table[config]
            for offset, is_correct in ((-0.19, True), (0.19, False)):
                chosen = values["predictive"][values["correct"] == is_correct]
                if len(chosen):
                    data.append(chosen)
                    positions.append(index + offset)
                    wrong.append(not is_correct)

        boxes = axis.boxplot(data, positions=positions, widths=0.32, patch_artist=True,
                             flierprops=dict(marker=".", markersize=3,
                                             markeredgecolor="#444444"))
        for patch, is_wrong in zip(boxes["boxes"], wrong):
            patch.set_facecolor("#9a9a9a" if is_wrong else "white")
            patch.set_edgecolor("black")
            patch.set_linewidth(0.7)
            if is_wrong:
                patch.set_hatch("////")
        for median in boxes["medians"]:
            median.set_color("black")

        axis.set_xticks(range(len(CONFIGS)))
        axis.set_xticklabels([CONFIG_TICKS[c] for c in CONFIGS])
        axis.set_ylim(-0.03, 1.03)
        axis.set_title(VARIANT_NAMES[variant])
        axis.grid(alpha=0.2, axis="y", linewidth=0.4)
    axes[0][0].set_ylabel("predictive entropy (bits)")

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black"),
               plt.Rectangle((0, 0), 1, 1, facecolor="#9a9a9a", edgecolor="black",
                             hatch="////")]
    fig.legend(handles, ["correct", "misclassified"], loc="lower center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, out_dir, name)


def decomposition(tables, out_dir, name):
    if not tables:
        return skip(name, "no probabilities.npz (run on the HPC with $WORK_DIR set)")

    fig, axes = plt.subplots(1, len(tables), figsize=(FULL_WIDTH, 2.4), sharey=True,
                             squeeze=False)
    for axis, (variant, table) in zip(axes[0], tables.items()):
        aleatoric = [table[c]["aleatoric"].mean() for c in CONFIGS]
        epistemic = [table[c]["epistemic"].mean() for c in CONFIGS]
        positions = np.arange(len(CONFIGS))
        axis.bar(positions, aleatoric, width=0.6, color="#d9d9d9", edgecolor="black",
                 linewidth=0.6, label="aleatoric")
        axis.bar(positions, epistemic, bottom=aleatoric, width=0.6, color="#4d4d4d",
                 edgecolor="black", linewidth=0.6, label="epistemic")
        for x, base, top in zip(positions, aleatoric, epistemic):
            axis.text(x, base + top + 0.015, f"{top:.3f}", ha="center", fontsize=6.8)
        axis.set_xticks(positions)
        axis.set_xticklabels([CONFIG_TICKS[c] for c in CONFIGS])
        axis.set_title(VARIANT_NAMES[variant])
        axis.grid(alpha=0.2, axis="y", linewidth=0.4)
    # Headroom for the epistemic values printed above the tallest bar.
    tallest = max(table[c]["aleatoric"].mean() + table[c]["epistemic"].mean()
                  for table in tables.values() for c in CONFIGS)
    axes[0][0].set_ylim(0, tallest * 1.15)
    axes[0][0].set_ylabel("mean entropy (bits)")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, out_dir, name)


def main():
    parser = argparse.ArgumentParser(description="Write the paper figures as PDF and PNG.")
    parser.add_argument("--work-dir", default=os.environ.get("WORK_DIR", ""),
                        help="Scratch folder holding uq_cv/ (needed for fig6 and fig7)")
    parser.add_argument("--uq-dir", default="uq_cv",
                        help="Pooled cross-validated uncertainty runs")
    parser.add_argument("--out-dir", default="figures/paper")
    args = parser.parse_args()

    repo_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"figures : {os.path.abspath(args.out_dir)}\n")

    cv = load_curves(os.path.join(repo_dir, "results", "evaluation_uq_cv_val.json"))
    test = load_curves(os.path.join(repo_dir, "results", "evaluation_test.json"))

    pipeline(args.out_dir)
    confusion(load_errors(os.path.join(repo_dir, "results", "evaluation_uq_cv_val.csv")),
              load_errors(os.path.join(repo_dir, "results", "evaluation_test.csv")),
              args.out_dir, "fig2_confusion")
    reliability(cv, args.out_dir, "fig3_reliability_cv")
    referral(cv, args.out_dir, "fig4_referral_cv")
    referral(test, args.out_dir, "fig5_referral_test")

    tables = load_uncertainty(args.work_dir, args.uq_dir, "val")
    entropy_correct_vs_wrong(tables, args.out_dir, "fig6_entropy_cv")
    decomposition(tables, args.out_dir, "fig7_decomposition_cv")


if __name__ == "__main__":
    main()
