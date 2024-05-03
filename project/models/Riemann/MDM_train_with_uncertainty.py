from moabb.datasets import BNCI2014_001
from pyriemann.estimation import Covariances
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_sample_weight
from sklearn.utils.extmath import softmax

from project.Utils.evaluate_and_plot import evaluate_uncertainty, plot_confusion_and_evaluate, plot_calibration, \
    find_best_temperature
from project.Utils.load_data import load_data
from project.models.Riemann.MDM_model_with_uncertainty import MDM

import numpy as np
from tqdm import tqdm

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)


def main():
    dataset = BNCI2014_001()
    n_classes = 4

    # datasets = [dataset]

    num_subjects = len(dataset.subject_list)
    for subject_id in tqdm(range(1, num_subjects + 1)):
        X, y, metadata = load_data(dataset, subject_id, n_classes)

        # Compute covariance matrices from the raw EEG signals
        cov_estimator = Covariances(estimator='lwf')
        X_cov = cov_estimator.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_cov, y, test_size=0.2, random_state=42)
        weights = compute_sample_weight('balanced', y=y_train)

        model = MDM(metric=dict(mean='riemann', distance='riemann'))

        model.fit(X_train, y_train, sample_weight=weights)

        # Predict the labels for the test set
        y_pred = model.predict(X_test)

        # Determine the confidence of the model
        distances = model.transform(X_test)
        temperature = find_best_temperature(y_pred, y_test, distances)

        prediction_proba = softmax(distances / temperature)

        confidence = np.max(prediction_proba, axis=1)

        # plot and evaluate
        plot_confusion_and_evaluate(y_pred, y_test, subject_id, save=True)
        evaluate_uncertainty(y_pred, y_test, confidence, subject_id)
        plot_calibration(y_pred, y_test, confidence, subject_id, save=True)


if __name__ == '__main__':
    main()
