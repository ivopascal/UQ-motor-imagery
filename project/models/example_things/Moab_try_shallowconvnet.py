from keras.optimizers import Adam

from Thesis.project.models.example_things.Moab_try_shallowmodel import ShallowConvNet
from Thesis.project.preprocessing.load_datafiles import read_data_moabb
from keras.utils import np_utils

from sklearn.preprocessing import LabelEncoder
import numpy as np



def main():
    
    X, y, metadata = read_data_moabb(1, 1, base_dir="../../../data/data_moabb_try/preprocessed")
    #print(y)

    # model = ShallowConvNet(nb_classes=4)
    # model.compile(loss='categorical_crossentropy', optimizer='adam')
    # fittedModel = model.fit(x, y)
    # predicted = model.predict(x, y)
    #

    # Reshape X to add the grayscale channel
    X_reshaped = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

    # Convert labels to categorical
    #y_categorical = np_utils.to_categorical(y)

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


    model = ShallowConvNet(nb_classes=num_unique_labels, Chans=22, Samples=1001, dropoutRate=0.5)

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_reshaped, y_categorical, epochs=100, batch_size=64, validation_split=0.2)

    model.save('../saved_trained_models/example/Moab_try_shallowconvnet.h5')

    #loaded_model = load_model('../saved_trained_models/example/Moab_try_shallowconvnet.h5')

if __name__ == '__main__':
    main()
