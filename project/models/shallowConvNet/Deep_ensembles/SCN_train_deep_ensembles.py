from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras_uncertainty.utils import entropy

from project.models.shallowConvNet.Deep_ensembles.SCNmodel import ShallowConvNet
from project.Utils.evaluate_and_plot import plot_confusion_and_evaluate, evaluate_uncertainty, plot_calibration

from tqdm import tqdm
import numpy as np

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)


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
    num_models = 2

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

            model = ShallowConvNet(nb_classes=4, Chans=22, Samples=1001, dropoutRate=0.5)
            optimizer = Adam(learning_rate=0.001)  # standard 0.001
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

            #weights = compute_sample_weight('balanced', y=y_train)
            model.fit(
                X_train,
                y_categorical,
                callbacks=[early_stopping],
                epochs=100, batch_size=64, validation_split=0.1, # sample_weight=weights,
                verbose=0,
            )

            predictions[model_idx] = model.predict(X_test)

        mean_predictions = np.mean(np.array([predictions]), axis=0)

        max_pred_0 = np.max(mean_predictions, axis=0)

        label_encoder = LabelEncoder()
        test_labels = label_encoder.fit_transform(y_test)

        predicted_classes = np.argmax(max_pred_0, axis=1)
        confidence = np.max(max_pred_0, axis=1)

        entr = entropy(test_labels, max_pred_0)
        print("Entropy: ", entr)

        # plot and evaluate
        plot_confusion_and_evaluate(predicted_classes, test_labels, subject_id, save=True)
        evaluate_uncertainty(predicted_classes, test_labels, confidence, subject_id)
        plot_calibration(predicted_classes, test_labels, confidence, subject_id, save=True)




if __name__ == '__main__':
    main()
