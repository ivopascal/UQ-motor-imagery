import numpy as np
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder

from Thesis.project.models.shallowConvNet.SCNmodel import ShallowConvNet
from Thesis.project.preprocessing.load_datafiles import read_data_moabb


def load_all_data(base_dir):
    X_all, y_all, groups = [], [], []
    num_subjects = 8  # Total number of subjects
    num_trials = 2    # Trials per subject

    for i in range(num_subjects):
        for j in range(num_trials):
            X, y, _ = read_data_moabb( i +1, j+ 1, base_dir=base_dir)
            X_all.append(X)
            y_all.append(y)
            groups.extend([i] * len(y))  # Use subject index as group label

    # Concatenate all data
    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    return X_all, y_all, np.array(groups)


def main():
    base_dir = "../../../data/data_moabb_try/preprocessed"
    X, y, groups = load_all_data(base_dir)

    model = ShallowConvNet(nb_classes=4, Chans=22, Samples=1001, dropoutRate=0.5)
    optimizer = Adam(learning_rate=0.005)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True)

    logo = LeaveOneGroupOut()
    fold = 1

    for train_index, test_index in logo.split(X, y, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Reshaping for the model
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

        label_encoder = LabelEncoder()
        y_train_integers = label_encoder.fit_transform(y_train)
        y_test_integers = label_encoder.transform(y_test)

        y_train_categorical = np_utils.to_categorical(y_train_integers, num_classes=4)
        y_test_categorical = np_utils.to_categorical(y_test_integers, num_classes=4)

        model.fit(
            X_train,
            y_train_categorical,
            validation_data=(X_test, y_test_categorical),
            callbacks=[early_stopping],
            epochs=100,
            batch_size=64
        )

        # Save the model for each fold if necessary
        model.save(f'../saved_trained_models/SCN/Crossval/SCN_MOABB_fold_{fold}.h5')
        fold += 1

    model.save('../saved_trained_models/SCN/SCN_CROSS.h5')


if __name__ == '__main__':
    main()