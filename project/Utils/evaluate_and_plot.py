from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils.extmath import softmax

from project.Utils import calibration

import seaborn as sns
import numpy as np

from project.Utils.calibration import get_ece


def plot_confusion_and_evaluate(y_pred, y_true, subject_id, save=True):
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Subject {subject_id} Validation accuracy: ", accuracy)

    f1 = f1_score(y_true, y_pred, average='macro')
    print(f'F1 score subject{subject_id}: ', f1)

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
    return


def evaluate_uncertainty(y_predictions, y_test, confidence, subject_id):
    overall_confidence = np.mean(confidence)
    print(f"Overall Confidence {subject_id}: ", overall_confidence)

    ece = calibration.get_ece(y_predictions, y_test, confidence)
    print(f"ECE {subject_id}: ", ece)
    mce = calibration.get_mce(y_predictions, y_test, confidence)
    print(f"MCE {subject_id}: ", mce)
    nce = calibration.get_nce(y_predictions, y_test, confidence)
    print(f"NCE {subject_id}: ", nce)


def plot_calibration(y_predictions, y_test, confidence, subject_id, save=True):
    calibration.plot_calibration_curve(y_predictions, y_test, confidence, subject_id, save)
    plt.clf()
    return


def find_best_temperature(predictions, y_test, distances):
    temperatures = np.linspace(0.1, 0.1, 2)  # search over these values
    best_ece = float('inf')
    best_temperature = 1.0
    for temp in temperatures:
        prediction_proba = softmax(distances / temp)
        confidence = np.max(prediction_proba, axis=1)
        ece = get_ece(predictions, y_test, confidence)
        if ece < best_ece:
            best_ece = ece
            best_temperature = temp
    return best_temperature
