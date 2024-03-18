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
import pickle

warnings.filterwarnings('ignore', category=FutureWarning)


def main():
    dataset = BNCI2014_001()
    dataset.subject_list = [9]
    sessions = dataset.get_data(subjects=dataset.subject_list)

    all_data = []
    all_labels = []

    subject_id = 9

    for session_name in ["0train", "1test"]:  # Iterate over sessions
        for run_id in range(5):  # Assuming 5 runs per session
            run_name = str(run_id)
            try:
                raw = dataset.get_data([subject_id])[subject_id][session_name][run_name]
                data, times = raw[:, :]
                events, event_ids = mne.events_from_annotations(raw)

                # Segment and label your data based on 'events'
                # Assuming each event marks the start of an epoch
                for event in events:
                    start = event[0]  # Start index of the event
                    end = start + 1000  # Example: end index, assuming a fixed epoch length
                    if end < data.shape[1]:  # Ensure the epoch does not exceed data bounds
                        epoch = data[:, start:end]
                        all_data.append(epoch)
                        all_labels.append(event[2])  # The label is the event ID

            except KeyError:
                # Handle the case where a session/run might not be available
                print(f"Data for subject {subject_id}, session {session_name}, run {run_name} not found.")

    # Convert lists to numpy arrays
    all_data = np.array(all_data)
    all_labels = np.array(all_labels)

    # Riemannian processing
    cov_estimator = Covariances(estimator='lwf')
    X_cov = cov_estimator.fit_transform(all_data)

    model_filename = 'riemannian_model.pkl'
    with open(model_filename, 'rb') as file:
        loaded_model = pickle.load(file)

    y_pred = loaded_model.predict(X_cov)
    accuracy = accuracy_score(all_labels, y_pred)
    print(f"Accuracy on subject 9: {accuracy}")


    #
    # # cv = KFold(n_splits=1, shuffle=True, random_state=42)
    # scores = cross_val_score(loaded_model, X_cov, all_labels, cv=2, n_jobs=1)
    #
    # label_encoder = LabelEncoder()
    # y_test_encoded = label_encoder.fit_transform(all_labels)
    #
    # # Now, y_test_encoded contains integer labels
    # class_counts = np.bincount(y_test_encoded)
    # most_frequent_class_proportion = np.max(class_counts) / len(y_test_encoded)
    #
    # print("MDM Classification accuracy: %f / Chance level: %f" % (np.mean(scores), most_frequent_class_proportion))


if __name__ == '__main__':
    main()
