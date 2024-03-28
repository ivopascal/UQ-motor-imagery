import numpy as np
from matplotlib import pyplot as plt
from moabb.datasets import BNCI2014_001
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import MotorImagery
from pyriemann.estimation import Covariances
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import make_pipeline
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import compute_class_weight, compute_sample_weight

from project.models.Riemann.MDM_model import MDM  # this is same to pyriemann
from project.preprocessing.load_datafiles import read_data_moabb
import warnings


warnings.filterwarnings('ignore', category=FutureWarning)


def main():
    dataset = BNCI2014_001()
    paradigm = MotorImagery(
        n_classes=4, fmin=7.5, fmax=30, tmin=0, tmax=None
    )

    # subjects = [1, 2, 3, 4, 5, 6, 7, 8]  # You can add more subjects
    # dataset.subject_list = subjects

    num_subjects = 9
    for subject_id in range(1, num_subjects + 1):
        subject = [subject_id]

        model = MDM(metric=dict(mean='riemann', distance='riemann'))

        X, y, metadata = paradigm.get_data(dataset=dataset, subjects=subject)

        # Compute covariance matrices from the raw EEG signals
        cov_estimator = Covariances(estimator='lwf')  # You can choose other estimators as well
        X_cov = cov_estimator.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_cov, y, test_size=0.2, random_state=42)
        weights = compute_sample_weight('balanced', y=y_train)

        model.fit(X_train, y_train, sample_weight=weights)

        # Predict the labels for the test set
        y_pred = model.predict(X_test)

        # Calculate and print the accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test accuracy for subject {subject_id}: {accuracy}")

        # y_probabilities = model.predict_proba(X_cov)
        # print("Probabilities: ", y_probabilities)



if __name__ == '__main__':
    main()
