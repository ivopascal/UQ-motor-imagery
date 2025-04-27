import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1)  Choose an uncertainty definition
# ------------------------------------------------------------------
def get_uncertainty(p, mode="uncertainty"):
    """
    Parameters
    ----------
    p : (N, C) ndarray – predicted class probabilities
    mode : "entropy" | "confidence" | "margin"
    Returns
    -------
    u : (N,) ndarray – larger  ⇒  more uncertain
    """
    if mode == "uncertainty":          # 1 − max‑probability, not 1 - for confidence
        return 1. - p.max(1)
    if mode == "confidence":
        return p.max(1)
    raise ValueError(f"unknown mode {mode}")


def accuracy_coverage_curve(y_true, predicted, uncertainty, n_steps=50):
    t_grid = np.linspace(1.0, 0.0, n_steps, endpoint=True)

    coverages, accuracies, thresholds = [], [], []

    num_samples = len(uncertainty)
    print("Number of samples:", num_samples)
    min_points = 5

    for threshold in t_grid:
        keep = uncertainty <= threshold                 # accept if uncertainty ≤ τ
        n_kept = keep.sum()
        print(f"n_kept: {n_kept}")
        cov    = n_kept / num_samples           # == keep.mean()
        print(f"cov: {cov}")

        if n_kept < min_points:       # skip: not enough data to estimate acc
            continue
        acc = accuracy_score(y_true[keep], predicted[keep].argmax(1))

        coverages.append(cov)
        accuracies.append(acc)
        thresholds.append(threshold)

    # by construction coverage decreases as τ decreases;
    # you can convert to rejection later:  rejection = 1 – coverage
    return np.asarray(coverages), np.asarray(accuracies), np.asarray(thresholds)

