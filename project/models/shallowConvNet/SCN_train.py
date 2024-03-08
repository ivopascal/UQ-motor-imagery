from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

from Thesis.project.models.shallowConvNet.SCNmodel import ShallowConvNet

from Thesis.project.preprocessing.load_datafiles import read_data_moabb
from keras.utils import np_utils

from sklearn.preprocessing import LabelEncoder
import numpy as np

from Thesis.project.preprocessing.load_datafiles_traintest import read_data_traintest


def main():

    model = ShallowConvNet(nb_classes=4, Chans=22, Samples=1001, dropoutRate=0.5)
    optimizer = Adam(learning_rate=0.005)    #standard 0.001
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,  # Number of epochs with no improvement
        mode='min',  # Minimize validation loss
        restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity
    )

    model.summary()

    # for i in range(5):
    #     for j in range(2):
    #         i = i+1
    #         j = j+1
    #X, y, metadata = read_data_moabb(i, j, base_dir="../../../data/data_moabb_try/preprocessed")

    X, y = read_data_traintest('train',  base_dir="../../../data/train_test_data/preprocessed")
    X_reshaped = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

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
    #model.predict(X_reshaped)

    model.save('../saved_trained_models/SCN/SCN_MOABB.h5')



if __name__ == '__main__':
    main()
