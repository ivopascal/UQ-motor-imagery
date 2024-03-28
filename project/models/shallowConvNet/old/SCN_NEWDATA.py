import numpy as np
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.utils import np_utils
from matplotlib import pyplot as plt
from moabb.datasets import BNCI2014_001
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import MotorImagery
from pyriemann.estimation import Covariances
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import make_pipeline
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import compute_class_weight, compute_sample_weight

from Thesis.project.models.Riemann.MDM_model import MDM  # this is same to pyriemann
from Thesis.project.models.shallowConvNet.SCNmodel import ShallowConvNet
from Thesis.project.preprocessing.load_datafiles import read_data_moabb
import warnings
import mne
import pickle


# warnings.filterwarnings('ignore', category=FutureWarning)

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
    plt.savefig('confusion.png')
    plt.show()


def loadData():
    dataset = BNCI2014_001()
    dataset.subject_list = [1, 2, 3, 4, 5, 6, 7, 8]
    sessions = dataset.get_data(subjects=dataset.subject_list)

    all_data = []
    all_labels = []

    for subject_id in dataset.subject_list:
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

    model = ShallowConvNet(nb_classes=4, Chans=26, Samples=250, dropoutRate=0.5)
    optimizer = Adam(learning_rate=0.005)  # standard 0.001
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,  # Number of epochs with no improvement
        mode='min',  # Minimize validation loss
        restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity
    )

    model.summary()

    #X_reshaped = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    X_reshaped = X.reshape(X.shape[0], 26, 250, 1)

    unique_labels = np.unique(y)
    num_unique_labels = len(unique_labels)
    assert num_unique_labels == 4, "The number of unique labels does not match the expected number of classes."

    label_encoder = LabelEncoder()
    y_integers = label_encoder.fit_transform(y)

    y_categorical = np_utils.to_categorical(y_integers, num_classes=num_unique_labels)

    model.fit(
        X_reshaped,
        y_categorical,
        callbacks=[early_stopping],
        epochs=100, batch_size=64, validation_split=0.2
    )

    predictions = model.predict(X_reshaped)
    predicted_classes = np.argmax(predictions, axis=1)

    # Since y was transformed to integers for encoding, use the same transformation for true labels
    true_classes = y_integers

    evaluate_model(predicted_classes, true_classes)

    model.save('../saved_trained_models/SCN/SCN_NEWDATA.h5')


if __name__ == '__main__':
    main()
