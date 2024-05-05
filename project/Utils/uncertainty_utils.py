from sklearn.utils.extmath import softmax

from project.Utils.calibration import get_ece

import numpy as np


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
