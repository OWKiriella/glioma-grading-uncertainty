# Uncertainty-Aware Deep Learning for Glioma Grading Using 3D MRI

Code for the MSc dissertation journal paper by Osadi Wathmavi Kiriella,
University of Plymouth, 2026.

The system grades a glioma from four 3D MRI scans as **HGG** (high grade) or **LGG** (low grade)
and estimates how far each grade can be trusted, so that uncertain patients can be referred for
expert review.

**Dataset:** BraTS2020, 369 patients (293 HGG, 76 LGG); scans T1, T1ce, T2 and FLAIR.
It is not included here and is available from the
[BraTS 2020 challenge](https://www.med.upenn.edu/cbica/brats2020/data.html).

## Pipeline

1. **Segmentation:** a 3D U-Net predicts the whole tumour.
2. **Grading:** a 3D CNN classifies HGG vs LGG from either
   - **whole-brain** input (the 4 scans), or
   - **mask-guided** input (the 4 scans plus the predicted tumour mask).
3. **Uncertainty:** softmax, Monte Carlo dropout (30 passes), deep ensemble (5 models), and
   the two combined.
4. **Evaluation:** balanced accuracy, AUC, calibration (ECE, Brier, NLL, temperature scaling)
   and uncertainty-based referral against a random baseline.

## Files, in the order they are run

| Step | File | What it does |
|---|---|---|
| 1 | `make_splits.py` | Splits patients 70/15/15 (train/val/test), stratified by grade |
| 1b | `make_folds.py` | Five cross-validation folds over the non-test patients |
| 2 | `prepare_data.py` | Preprocesses every scan once (orientation, crop, 1.5 mm, 128×128×96, z-score) |
| 2b | `check_cache.py` | Checks the preprocessed data |
| 3 | `train_seg.py` | Trains the 3D U-Net |
| 4 | `predict_seg.py` | Predicts a tumour mask for every patient |
| 5 | `train_grader.py` | Trains the grading CNN (run with `--seed 1` to `5` for the ensemble) |
| 6 | `predict_mc.py` | Predictions and uncertainty for all four configurations |
| 7 | `pool_folds.py` | Joins the five cross-validation folds |
| 8 | `evaluate.py` | Calibration, referral and bootstrap confidence intervals |
| 9 | `make_paper_figures.py` | Draws the figures in the paper |
| – | `folds.py` | Shared helper used by the scripts above |

Each script explains its own options: `python <script>.py --help`.
The training scripts also take `--smoke-test`, a tiny run on CPU to check the code works.

## Requirements

Python 3.11, a CUDA GPU for training, and:

```bash
pip install torch
pip install -r requirements.txt
```
