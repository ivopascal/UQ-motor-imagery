from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_sample_weight

from project.models.shallowConvNet.DUQ.SCN_model_DUQ import ShallowConvNet

from keras.utils import np_utils

from sklearn.preprocessing import LabelEncoder
import numpy as np

import seaborn as sns
from sklearn.utils.extmath import softmax

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
    dataset = BNCI2014_001()        # load dataset
    paradigm = MotorImagery(        # make paradigm, filter between 7.5 and 30 Hz
        n_classes=4, fmin=7.5, fmax=30, tmin=0, tmax=None
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,  # Number of epochs with no improvement
        mode='min',  # Minimize validation loss
        restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity
    )

    num_subjects = 9
    for subject_id in range(1, num_subjects + 1):       # loop to take data and make model per subject
        subject = [subject_id]

        X, y, metadata = paradigm.get_data(dataset=dataset, subjects=subject)       # get the data for specific subject

        unique_labels = np.unique(y)
        num_unique_labels = len(unique_labels)
        assert num_unique_labels == 4, "The number of unique labels does not match the expected number of classes."

        X_reshaped = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

        X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

        # make the labels categorical
        label_encoder = LabelEncoder()
        y_integers = label_encoder.fit_transform(y_train)
        y_categorical = np_utils.to_categorical(y_integers, num_classes=num_unique_labels)

        net = ShallowConvNet()
        model = net.build()

        # weights = compute_sample_weight('balanced', y=y_train)

        model.fit(      # train the model
            X_train,
            y_categorical,
            callbacks=[early_stopping],     # early stopping seems to work worse with DUQ, it needs long to train it seems
            epochs=200, batch_size=64, validation_split=0.1 #, sample_weight=weights
            ,verbose=1,
        )

        # model.save(f'../saved_trained_models/SCN/PerSubject/subject{subject_id}')

        # Now test how good the model performs
        label_encoder = LabelEncoder()
        test_labels = label_encoder.fit_transform(y_test)

        predictions = model.predict(X_test)

        print("Predictions are: ", predictions)

        predicted_classes = np.argmax(predictions, axis=1)      # slides matthias: arg max Kc(fθ(x), ec )

        # print("Predicted classes are: ", predicted_classes)

        # todo this below gives okay results but it is not a real probability as in the riemann as it does not sum to 1
        # confidence = np.max(predictions, axis=1)
        # print("Confidence: ", confidence)
        # overall_confidence = confidence.mean()
        # print("Overall confidence: ", overall_confidence)

        prediction_proba = softmax(-predictions ** 2)
        print("Prediction probabilities: ", prediction_proba)
        # this is done the same as with the Riemann model (softmax of negative squared distances.),
        #    however the probabilities do not really make sense if done this way, so for example :
        # [6.78250417e-02 6.87640011e-02 4.48920007e-04 9.96227741e-01] these distances would be
        # [0.29613385 0.29609588 0.2974992  0.1102711 ] these probabilities

        confidence = np.max(prediction_proba, axis=1)
        print("Confidence: ", confidence)

        overall_confidence = np.mean(confidence)
        print("Overall Confidence: ", overall_confidence)

        evaluate_model(predicted_classes, test_labels, subject_id)




if __name__ == '__main__':
    main()
