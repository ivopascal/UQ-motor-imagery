import numpy as np
from matplotlib import pyplot as plt
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from pyriemann.estimation import Covariances
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.utils import compute_sample_weight

from project.models.Riemann.MDM_model_with_uncertainty import MDM  # this is same to pyriemann
import warnings


warnings.filterwarnings('ignore', category=FutureWarning)


def evaluate_model(y_pred, y_true, subject_id):
    subject_id = subject_id
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Subject {subject_id} Validation accuracy: ", accuracy)

    f1 = f1_score(y_true, y_pred, average='macro')
    print(f'F1 score subject{subject_id}: \n', f1)

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"Confusion Matrix subject {subject_id}")
    # plt.savefig(f"./graphs/confusion_subject{subject_id}.png")
    plt.show()


def main():
    dataset = BNCI2014_001()
    paradigm = MotorImagery(
        n_classes=4, fmin=7.5, fmax=30, tmin=0, tmax=None
    )

    num_subjects = 9
    for subject_id in range(1, num_subjects + 1):
        subject = [subject_id]

        model = MDM(metric=dict(mean='riemann', distance='riemann'))

        X, y, metadata = paradigm.get_data(dataset=dataset, subjects=subject)

        # Compute covariance matrices from the raw EEG signals
        cov_estimator = Covariances(estimator='lwf')
        X_cov = cov_estimator.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_cov, y, test_size=0.2, random_state=42)
        weights = compute_sample_weight('balanced', y=y_train)

        model.fit(X_train, y_train, sample_weight=weights)

        # Predict the labels for the test set
        y_pred = model.predict(X_test)
        evaluate_model(y_pred, y_test, subject_id)

        # Determine the confidence of the model
        # prediction_proba = model.predict_proba(X_test)          # dit geeft net aan iets betere waardes maar weinig verschil
        prediction_proba = model.predict_proba_temperature(X_test, 0.2)

        confidence = np.max(prediction_proba, axis=1)
        overall_confidence = np.mean(confidence)
        print(f"Overall Confidence: {overall_confidence}")



if __name__ == '__main__':
    main()
