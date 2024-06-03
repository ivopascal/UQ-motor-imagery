from moabb.datasets import BNCI2014_001, BNCI2015_004, AlexMI, Zhou2016, BNCI2014_004, Schirrmeister2017, PhysionetMI, \
    BNCI2014_002
from moabb.paradigms import MotorImagery
from pyriemann.estimation import Covariances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import compute_sample_weight
from sklearn.utils.extmath import softmax

from project.Utils.evaluate_and_plot import evaluate_uncertainty, plot_confusion_and_evaluate, plot_calibration
from project.Utils.load_data import load_data
from project.Utils.uncertainty_utils import find_best_temperature
from project.models.Riemann.MDM_model_with_uncertainty import MDM

import numpy as np
from tqdm import tqdm

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)


def main():
    dataset1 = BNCI2014_002()
    dataset2 = Zhou2016()
    dataset3 = BNCI2014_004()
    dataset4 = BNCI2014_001()       # original one

    datasets = [dataset1, dataset2, dataset3, dataset4]

    n_classes = [2, 3, 2, 4]

    # This unfortunately cannot really be done more elegantly, because the paradigm to get the data needs
    #   the number of classes, and the dataset not the dict of get_data can get the number of classes

    for dataset, num_class in zip(datasets, n_classes):
        num_subjects = len(dataset.subject_list)
        for subject_id in range(1, num_subjects + 1):
            dataset_id = datasets.index(dataset) + 1

            X, y, metadata = load_data(dataset, subject_id, num_class)

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

            # confidence = np.max(prediction_proba, axis=1)

            label_encoder = LabelEncoder()
            test_labels = label_encoder.fit_transform(y_test)
            predictions = label_encoder.fit_transform(y_pred)

            # plot and evaluate
            plot_confusion_and_evaluate(predictions, test_labels,
                                        subject_id= subject_id, dataset_id=dataset_id, save=False)

            evaluate_uncertainty(predictions, test_labels, prediction_proba,
                                 subject_id=subject_id, dataset_id=dataset_id, save=False)

            plot_calibration(predictions, test_labels, prediction_proba,
                             subject_id=subject_id, dataset_id=dataset_id, save=False)


if __name__ == '__main__':
    main()
