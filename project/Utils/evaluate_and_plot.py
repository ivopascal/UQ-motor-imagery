from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from project.Utils import calibration

import seaborn as sns
import numpy as np


def plot_confusion_and_evaluate(y_pred, y_true, subject_id, save=True):
    f = open(f"./results/evaluation_subject{subject_id}.txt", "w")

    accuracy = accuracy_score(y_true, y_pred)
    f.write(f"Subject {subject_id} Validation accuracy: {accuracy}\n")

    f1 = f1_score(y_true, y_pred, average='macro')
    f.write(f'F1 score subject{subject_id}: {f1}\n')

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"Confusion Matrix subject {subject_id}")
    if save:
        plt.savefig(f"./graphs/confusion_plots/confusion_subject{subject_id}.png")
    # else:
    plt.show()
    plt.clf()

    f.close()
    return


def evaluate_uncertainty(y_predictions, y_test, confidence, subject_id):
    f = open(f"./results/evaluation_subject{subject_id}.txt", "a")

    overall_confidence = np.mean(confidence)
    f.write(f"Overall Confidence {subject_id}: {overall_confidence}\n")

    ece = calibration.get_ece(y_predictions, y_test, confidence)
    f.write(f"ECE {subject_id}: {ece}\n")

    mce = calibration.get_mce(y_predictions, y_test, confidence)
    f.write(f"MCE {subject_id}: {mce}\n")

    nce = calibration.get_nce(y_predictions, y_test, confidence)
    f.write(f"NCE {subject_id}: {nce}\n")


def plot_calibration(y_predictions, y_test, confidence, subject_id, save=True):
    calibration.plot_calibration_curve(y_predictions, y_test, confidence, subject_id, save)
    plt.clf()
    return

