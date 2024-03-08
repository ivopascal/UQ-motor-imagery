import numpy as np
from keras.models import load_model
from keras.utils import np_utils
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from Thesis.project.preprocessing.load_datafiles import read_data_moabb

from keras import backend as K
import seaborn as sns


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


def main():
    loaded_model = load_model('../saved_trained_models/SCN/SCN_MOABB.h5',
                              custom_objects={'square': square, 'log': log})

    for i in range(5, 8):
        for j in range(2):
            i = i + 1
            j = j + 1

            X, y, metadata = read_data_moabb(i, j, base_dir="../../../data/data_moabb_try/preprocessed")
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


if __name__ == '__main__':
    main()
