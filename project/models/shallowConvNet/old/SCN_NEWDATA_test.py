import mne
import numpy as np
from keras.models import load_model
from keras.utils import np_utils
from matplotlib import pyplot as plt
from moabb.datasets import BNCI2014_001
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from Thesis.project.preprocessing.load_datafiles import read_data_moabb

from keras.models import Model

from keras import backend as K
import seaborn as sns

from Thesis.project.preprocessing.load_datafiles_traintest import read_data_traintest


def square(x):
    return K.square(x)


def log(x):
    return K.log(K.clip(x, min_value=1e-7, max_value=10000))


def evaluate_model(y_pred, y_true):
    accuracy = accuracy_score(y_true, y_pred)
    print("Validation accuracy: ", accuracy)

    f1 = f1_score(y_true, y_pred, average='macro')
    print('F1 score: ', f1)

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    #plt.savefig('confusion.png')
    plt.show()


def loadData():
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
                    end = start + 250  # Example: end index, assuming a fixed epoch length
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

    return all_data, all_labels

def main():
    X, y = loadData()
    print("X shape:", X.shape)
    print("Y shape: ", y.shape)

    loaded_model = load_model('../saved_trained_models/SCN/SCN_NEWDATA.h5',
                              custom_objects={'square': square, 'log': log})

    X_reshaped = X.reshape(X.shape[0], 26, 250, 1)

    unique_labels = np.unique(y)
    num_unique_labels = len(unique_labels)
    assert num_unique_labels == 4, "The number of unique labels does not match the expected number of classes."

    label_encoder = LabelEncoder()
    y_integers = label_encoder.fit_transform(y)

    y_categorical = np_utils.to_categorical(y_integers, num_classes=num_unique_labels)

    predictions = loaded_model.predict(X_reshaped)
    predicted_classes = np.argmax(predictions, axis=1)

    # Since y was transformed to integers for encoding, use the same transformation for true labels
    true_classes = y_integers

    evaluate_model(predicted_classes, true_classes)


if __name__ == '__main__':
    main()
