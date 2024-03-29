from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_sample_weight
from tqdm import tqdm

from project.models.shallowConvNet.SCNmodel import ShallowConvNet

from project.preprocessing.load_datafiles import read_data_moabb
from keras.utils import np_utils

from sklearn.preprocessing import LabelEncoder
import numpy as np

from project.preprocessing.load_datafiles_traintest import read_data_traintest

import seaborn as sns

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)


def evaluate_model(y_pred, y_true, subject_id):
    subject_id = subject_id
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Subject {subject_id} Validation accuracy: ", accuracy)

    f1 = f1_score(y_true, y_pred, average='macro')
    print(f'F1 score subject{subject_id}: ', f1)

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"Confusion Matrix subject {subject_id}")
    #plt.savefig('confusion.png')
    plt.show()


def main():
    dataset = BNCI2014_001()
    paradigm = MotorImagery(
        n_classes=4, fmin=7.5, fmax=30, tmin=0, tmax=None
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=7,  # Number of epochs with no improvement
        mode='min',  # Minimize validation loss
        restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity
    )

    num_subjects = 9
    num_models = 5

    for subject_id in range(1, num_subjects + 1):
        subject = [subject_id]

        X, y, metadata = paradigm.get_data(dataset=dataset, subjects=subject)

        unique_labels = np.unique(y)
        num_unique_labels = len(unique_labels)
        assert num_unique_labels == 4, "The number of unique labels does not match the expected number of classes."

        X_reshaped = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

        X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

        label_encoder = LabelEncoder()
        y_integers = label_encoder.fit_transform(y_train)
        y_categorical = np_utils.to_categorical(y_integers, num_classes=num_unique_labels)

        predictions = np.zeros((num_models, X_test.shape[0], len(np.unique(y))))
        for model_idx in tqdm(range(num_models)):

            # make a model for every individual subject
            model = ShallowConvNet(nb_classes=4, Chans=22, Samples=1001, dropoutRate=0.5)
            optimizer = Adam(learning_rate=0.001)  # standard 0.001
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

            weights = compute_sample_weight('balanced', y=y_train)

            model.fit(
                X_train,
                y_categorical,
                callbacks=[early_stopping],
                epochs=100, batch_size=64, validation_split=0.1 #, sample_weight=weights
                ,verbose=0,
            )

            predictions[model_idx] = model.predict(X_test)

        # predictions = model.predict(X_test)
        # predicted_classes = np.argmax(predictions, axis=1)

        label_encoder = LabelEncoder()
        test_labels = label_encoder.fit_transform(y_test)

        mean_predictions = np.mean(predictions, axis=0)
        uncertainty = np.var(predictions, axis=0)

        y_pred = np.argmax(mean_predictions, axis=1)
        # accuracy = accuracy_score(y_test, y_pred)
        # print(f"Test accuracy for subject {subject_id}: {accuracy}")
        # print(f"Prediction uncertainty: {np.mean(uncertainty)}")

        # Calculate and print the accuracy
        #accuracy = accuracy_score(test_labels, predicted_classes)
        #print(f"Test accuracy for subject {subject_id}: {accuracy}")
        print(f"Prediction uncertainty: {np.mean(uncertainty)}")

        evaluate_model(y_pred, test_labels, subject_id)



if __name__ == '__main__':
    main()
