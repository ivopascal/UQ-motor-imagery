import numpy as np
from keras.models import load_model
from keras.utils import np_utils
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from keras import backend as K
import seaborn as sns

from Thesis.project.preprocessing.load_datafiles import read_data_moabb
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
    plt.show()


def evaluate_folds():
    num_subjects = 8  # Adjust based on your cross-validation setup
    # Loop through each fold's model and evaluate it
    for i in range(1, num_subjects + 1):
        model_path = f'../saved_trained_models/SCN/Crossval/SCN_MOABB_fold_{i}.h5'
        loaded_model = load_model(model_path, custom_objects={'square': square, 'log': log})

        # Load the corresponding test data for the fold
        # Assuming you have a mechanism to load test data for each fold (adjust as necessary)
        X, y, _ = read_data_moabb(subject_id=i, trial_id=2, base_dir="../../../data/data_moabb_try/preprocessed")
        X_reshaped = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

        label_encoder = LabelEncoder()
        y_integers = label_encoder.fit_transform(y)
        y_categorical = np_utils.to_categorical(y_integers)

        predictions = loaded_model.predict(X_reshaped)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = y_integers

        print(f"Evaluating fold {i}")
        evaluate_model(predicted_classes, true_classes)


def evaluate_overall_model():
    loaded_model = load_model('../saved_trained_models/SCN/SCN_CROSS.h5',
                              custom_objects={'square': square, 'log': log})

    X, y = read_data_traintest('test', base_dir="../../../data/train_test_data/preprocessed")
    #X, y, metadata = read_data_moabb(9, 2, base_dir="../../../data/data_moabb_try/preprocessed")

    X_reshaped = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

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

def main():
    #evaluate_folds()
    evaluate_overall_model()


if __name__ == '__main__':
    main()
