import numpy as np
from keras.models import load_model
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

from Thesis.project.preprocessing.load_datafiles import read_data_moabb

from keras import backend as K


def square(x):
    return K.square(x)


def log(x):
    return K.log(K.clip(x, min_value=1e-7, max_value=10000))


def main():
    X, y, metadata = read_data_moabb(8, 1, base_dir="../../../../data/data_moabb_try/preprocessed")

    X_reshaped = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

    label_encoder = LabelEncoder()
    y_integers = label_encoder.fit_transform(y)
    y_categorical = np_utils.to_categorical(y_integers, num_classes=np.unique(y_integers).size)

    loaded_model = load_model('../saved_trained_models/example/SCN_MOABB.h5',
                              custom_objects={'square': square, 'log': log})

    loss, accuracy = loaded_model.evaluate(X_reshaped, y_categorical)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")

    # For predictions, use the predict method
    predictions = loaded_model.predict(X_reshaped)
    # Convert predictions to label indices if necessary
    predicted_classes = np.argmax(predictions, axis=1)

    predicted_labels = label_encoder.inverse_transform(predicted_classes)


if __name__ == '__main__':
    main()
