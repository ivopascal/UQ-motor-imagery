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
    X, y, metadata = read_data_moabb(9, 2, base_dir="../../../data/data_moabb_try/preprocessed")

    X_reshaped = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)


    unique_labels = np.unique(y)
    num_unique_labels = len(unique_labels)
    print(f"Number of unique labels: {num_unique_labels}")

    # Ensure this matches nb_classes in your model
    assert num_unique_labels == 4, "The number of unique labels does not match the expected number of classes."

    # Encode labels to integers
    label_encoder = LabelEncoder()
    y_integers = label_encoder.fit_transform(y)

    # Convert integer labels to one-hot encoding
    y_categorical = np_utils.to_categorical(y_integers, num_classes=num_unique_labels)

    loaded_model = load_model('../saved_trained_models/example/Moab_try_shallowconvnet.h5',
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
