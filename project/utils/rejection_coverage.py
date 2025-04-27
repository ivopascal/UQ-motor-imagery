import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1)  Choose an uncertainty definition
# ------------------------------------------------------------------
def get_uncertainty(p, mode="confidence"):
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
    raise ValueError(f"unknown mode {mode}")


def accuracy_coverage_curve(y_true, predicted, uncertainty, n_steps=50):
    """
    Returns
    -------
    coverages : fraction of samples kept
    accuracies     : accuracy among kept samples
    thresholds: corresponding uncertainty threshold
    """
    # Go from least uncertainty (so least confident) to most uncertain predictions,
    # this way we reject atleast 1 sample and at most all samples, with n_steps sample in between
    # this does not guarantee that every step has a different number of datapoints that fall in the range of uncertainty

    t_grid = np.linspace(uncertainty.min(), uncertainty.max(), n_steps, endpoint=True)
    coverages, accuracies = [], []

    for threshold in t_grid:
        # keep all samples that have less uncertainty (more confidence) than the threshold
        # so we reject samples with too high uncertainty
        keep = uncertainty <= threshold

        n_kept = keep.sum()
        min_points = 5

        if n_kept < min_points:  # if too little datapoints, continue
            continue

        cov  = keep.mean()

        if cov == 0.0:                 # if we reject all then skip
            continue
        acc  = accuracy_score(y_true[keep], predicted[keep].argmax(1))
        coverages.append(cov)
        accuracies.append(acc)   #1 - acc for risk

    return np.asarray(coverages), np.asarray(accuracies), t_grid[:len(coverages)]

