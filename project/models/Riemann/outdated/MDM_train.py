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

from Thesis.project.models.Riemann.MDM_model import MDM  # this is same to pyriemann
from Thesis.project.preprocessing.load_datafiles import read_data_moabb
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)


def printLabels(y_train, y_test):
    print("Yshape TRAINING: ", y_train.shape)
    print("left_hand:", sum(y_train == 'left_hand'))
    print("right_hand:", sum(y_train == 'right_hand'))
    print("Feet:", sum(y_train == 'feet'))
    print("tongue:", sum(y_train == 'tongue'))

    print("Yshape TESTING: ", y_test.shape)
    print("left_hand:", sum(y_test == 'left_hand'))
    print("right_hand:", sum(y_test == 'right_hand'))
    print("Feet:", sum(y_test == 'feet'))
    print("tongue:", sum(y_test == 'tongue'))


def main():
    dataset = BNCI2014_001()
    paradigm = MotorImagery(
        n_classes=4, fmin=7.5, fmax=30, tmin=0, tmax=None
    )

    subjects = [1, 2, 3, 4, 5, 6, 7, 8]  # You can add more subjects
    dataset.subject_list = subjects
    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=subjects)


    # Compute covariance matrices from the raw EEG signals
    cov_estimator = Covariances(estimator='lwf')  # You can choose other estimators as well
    X_cov = cov_estimator.fit_transform(X)

    print("Shape X:", X_cov.shape, "Shape y:", y.shape)
    # This are the dimensions before covariance: Shape X: (4608, 22, 1001) Shape y: (4608,) after covariance 4608,22,22

    model = MDM(metric=dict(mean='riemann', distance='riemann'))

    X_train, X_test, y_train, y_test = train_test_split(X_cov, y, test_size=0.2, random_state=42)

    #printLabels(y_train, y_test)

    weights = compute_sample_weight('balanced', y=y_train)

    # model.fit(X_train, y_train, sample_weight=weights)
    #
    # # Predict the labels for the test set
    # y_pred = model.predict(X_test)
    #
    # # Calculate and print the accuracy
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f"Accuracy: {accuracy}")

    # scores = cross_val_score(model, X_cov, y, cv=8, scoring='accuracy')
    #
    # # Print the accuracy for each fold and the average accuracy
    # print(f"Cross-Validation Accuracy Scores: {scores}")
    # print(f"Average Cross-Validation Accuracy: {np.mean(scores)}")

    cv = KFold(n_splits=8, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_test, y_test, cv=cv, n_jobs=1)

    # # Printing the results
    # class_balance = np.mean(y_test == y_test[0])
    # class_balance = max(class_balance, 1. - class_balance)
    # print("MDM Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
    #                                                               class_balance))
    label_encoder = LabelEncoder()
    y_test_encoded = label_encoder.fit_transform(y_test)

    # Now, y_test_encoded contains integer labels
    class_counts = np.bincount(y_test_encoded)
    most_frequent_class_proportion = np.max(class_counts) / len(y_test_encoded)

    print("MDM Classification accuracy: %f / Chance level: %f" % (np.mean(scores), most_frequent_class_proportion))

if __name__ == '__main__':
    main()
