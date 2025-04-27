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

    t_grid = np.linspace(uncertainty.min(), 1.0, n_steps, endpoint=True)
    coverages, accuracies = [], []

    print("Uncertainty: ", uncertainty)
    print("number of samples: ", len(uncertainty))

    for threshold in t_grid:
        # keep all samples that have less uncertainty (more confidence) than the threshold
        # so we reject samples with too high uncertainty
        keep = uncertainty <= threshold
        print("threshold", threshold)

        n_kept = keep.sum()
        print("n_kept", n_kept)
        min_points = 5

        cov  = keep.mean()
        print("cov", cov)

        if cov == 0:                 # if we reject all then skip
            # acc = 1.0
            print("We rejected all, accuracy = 1")
            continue
        else:
            if n_kept < min_points:  # if too little datapoints, continue
                print("Too little datapoints, threshold was: ", threshold)
                continue
            acc = accuracy_score(y_true[keep], predicted[keep].argmax(1))

        coverages.append(cov)
        accuracies.append(acc)   #1 - acc for risk

    return np.asarray(coverages), np.asarray(accuracies), t_grid[:len(coverages)]

