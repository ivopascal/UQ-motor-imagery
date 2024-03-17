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
import mne

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


# def main():
#     dataset = BNCI2014_001()
#
#     dataset.subject_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
#
#     sessions = dataset.get_data(subjects=[1, 2, 3, 4, 5, 6, 7, 8, 9])
#
#     session_name = "0train"
#     run_name = "0"
#     raw = sessions[1][session_name][run_name]
#
#     # print(raw)
#
#     data, times = raw[:, :]  # data: numpy array of shape (n_channels, n_times)
#     events, event_ids = mne.events_from_annotations(raw)
#
#
#     # Compute covariance matrices from the raw EEG signals
#     cov_estimator = Covariances(estimator='lwf')  # You can choose other estimators as well
#     X_cov = cov_estimator.fit_transform(data)
#
#     print("Shape X:", X_cov.shape, "Shape y:", y.shape)
#     # This are the dimensions before covariance: Shape X: (4608, 22, 1001) Shape y: (4608,) after covariance 4608,22,22
#
#     model = MDM(metric=dict(mean='riemann', distance='riemann'))
#
#     X_train, X_test, y_train, y_test = train_test_split(X_cov, events, test_size=0.2, random_state=42)
#
#
#     cv = KFold(n_splits=8, shuffle=True, random_state=42)
#     scores = cross_val_score(model, X_test, y_test, cv=cv, n_jobs=1)
#
#     label_encoder = LabelEncoder()
#     y_test_encoded = label_encoder.fit_transform(y_test)
#
#     # Now, y_test_encoded contains integer labels
#     class_counts = np.bincount(y_test_encoded)
#     most_frequent_class_proportion = np.max(class_counts) / len(y_test_encoded)
#
#     print("MDM Classification accuracy: %f / Chance level: %f" % (np.mean(scores), most_frequent_class_proportion))

def main():
    dataset = BNCI2014_001()
    dataset.subject_list = [1, 2, 3, 4, 5, 6, 7, 8]
    sessions = dataset.get_data(subjects=dataset.subject_list)

    all_data = []
    all_labels = []

    for subject_id in range(1, len(dataset.subject_list) + 1):
        session_name = "0train"
        run_name = "0"
        raw = sessions[subject_id][session_name][run_name]

        # Extract data and events
        data, times = raw[:, :]  # data: numpy array of shape (n_channels, n_times)
        events, event_ids = mne.events_from_annotations(raw)

        # Here you would segment the data based on events and extract epochs
        # For now, we will assume that 'data' is already in the form of epochs
        # In practice, you should extract epochs based on 'events'

        # Append data and labels (assuming 'data' is segmented into epochs)
        # and 'event_ids' map directly to the labels for each epoch
        for event in events:
            epoch = data[:, event[0]:event[0] + 1000]  # Example: epoch window of 1000 samples
            all_data.append(epoch)
            all_labels.append(event[2])  # Assuming event[2] is the label

    # Convert lists to numpy arrays
    all_data = np.array(all_data)
    all_labels = np.array(all_labels)

    # Riemannian processing
    cov_estimator = Covariances(estimator='lwf')
    X_cov = cov_estimator.fit_transform(all_data)

    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X_cov, all_labels, test_size=0.2, random_state=42)

    # Model fitting
    # model = MDM(metric=dict(mean='riemann', distance='riemann'))
    # model.fit(X_train, y_train)
    #
    # # Prediction and evaluation
    # y_pred = model.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f"Accuracy: {accuracy}")

    model = MDM(metric=dict(mean='riemann', distance='riemann'))

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
