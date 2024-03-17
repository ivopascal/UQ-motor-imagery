import numpy as np
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder

from Thesis.project.preprocessing.load_datafiles import read_data_moabb

from SCN_hypermodel import KerasShallowConvNet
import keras_tuner as kt


def load_all_data(base_dir):
    X_all, y_all, groups = [], [], []
    num_subjects = 8  # Total number of subjects
    num_trials = 2  # Trials per subject

    for i in range(num_subjects):
        for j in range(num_trials):
            X, y, _ = read_data_moabb(i + 1, j + 1, base_dir=base_dir)
            X_all.append(X)
            y_all.append(y)
            groups.extend([i] * len(y))  # Use subject index as group label

    # Concatenate all data
    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    return X_all, y_all, np.array(groups)


def tune_model(X_train, X_test, y_train, y_test):
    def model_builder(hp):
        return KerasShallowConvNet(hp)

    tuner = kt.Hyperband(
        model_builder,
        objective='val_accuracy',
        max_epochs=4,
        hyperband_iterations=2,
        directory='hyperband',
        project_name='SCN_hyperparams',
        overwrite=True
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True)

    tuner.search(X_train, y_train,
                 epochs=10,
                 validation_data=(X_test, y_test),
                 callbacks=[early_stopping])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    return best_hps


def main():
    base_dir = "../../../data/data_moabb_try/preprocessed"
    X, y, groups = load_all_data(base_dir)


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

        best_hps = tune_model()

        # Print the best hyperparameters
        print(
            f"Best hyperparameters:\n- GRU Units: {best_hps.get('gru_units')}\n- Optimizer: {best_hps.get('optimizer')}\n- Learning Rate: {best_hps.get('learning_rate')}")

        # Build the model with the best hyperparameters
        model = build_model(best_hps, tokenizer, word2vec_model, embedding_matrix, texts_tokenized)

        # Compile and summarize the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.summary()


if __name__ == '__main__':
    main()
