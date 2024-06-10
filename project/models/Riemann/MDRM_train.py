import pandas as pd
from matplotlib import pyplot as plt
from moabb.datasets import BNCI2014_001, Zhou2016, BNCI2014_004, BNCI2014_002
from pyriemann.estimation import Covariances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.utils import compute_sample_weight
from sklearn.utils.extmath import softmax

from project.Utils.evaluate_and_plot import evaluate_uncertainty, plot_confusion_and_evaluate, plot_calibration
from project.Utils.load_data import load_data
from project.Utils.uncertainty_utils import find_best_temperature
from project.models.Riemann.MDRM_model import MDM

import seaborn as sns

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)


def plot_covariance_matrices(mdm, num_classes):
    """
    Plots the covariance matrices for all classes.

    Parameters:
    mdm : object
        The MDM object that contains the fitted covariance matrices.
    epochs : object
        The epochs object that contains the channel names.
    num_classes : int
        The number of classes to plot the covariance matrices for.
    """
    fig, axes = plt.subplots(1, num_classes, figsize=(4 * num_classes, 4))

    if num_classes == 1:
        axes = [axes]

    for i in range(num_classes):
        df = pd.DataFrame(data=mdm.covmeans_[i])
        ax = axes[i]
        g = sns.heatmap(df, ax=ax, square=True, cbar=False, xticklabels=2, yticklabels=2)
        g.set_title(f'Mean covariance - class {i + 1}')
        ax.set_xticklabels(ax.get_xticklabels(), rotation='vertical')
        ax.set_yticklabels(ax.get_yticklabels(), rotation='horizontal')

    plt.tight_layout()
    plt.show()

def main():
    dataset1 = BNCI2014_002()
    dataset2 = Zhou2016()
    dataset3 = BNCI2014_004()
    dataset4 = BNCI2014_001()

    datasets = [dataset1, dataset2, dataset3, dataset4]

    n_classes = [2, 3, 2, 4]

    # This unfortunately cannot really be done more elegantly, because the paradigm to get the data needs
    #   the number of classes, and the dataset nor the dict of get_data can get the number of classes

    for dataset, num_class in zip(datasets, n_classes):
        num_subjects = len(dataset.subject_list)
        for subject_id in range(1, num_subjects + 1):
            dataset_id = datasets.index(dataset) + 1

            X, y, metadata = load_data(dataset, subject_id, num_class)

            cov_estimator = Covariances(estimator='lwf')
            X_cov = cov_estimator.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X_cov, y, test_size=0.2, random_state=42)
            weights = compute_sample_weight('balanced', y=y_train)

            model = MDM(metric=dict(mean='riemann', distance='riemann'))

            model.fit(X_train, y_train, sample_weight=weights)

            plot_covariance_matrices(model, num_class)

            # Predict the labels for the test set
            y_pred = model.predict(X_test)

            # Determine the confidence of the model
            distance_pred = model.transform(X_test)
            distances = normalize(distance_pred, axis=1, norm='l1')
            temperature = find_best_temperature(y_pred, y_test, distances)

            prediction_proba = softmax(distances / temperature)

            # prediction_proba = model.predict_proba(X_test)        # alternative manner to get probabilities
            # confidence = np.max(prediction_proba, axis=1)

            label_encoder = LabelEncoder()
            predictions = label_encoder.fit_transform(y_pred)
            test_labels = label_encoder.fit_transform(y_test)

            # plot and evaluate
            plot_confusion_and_evaluate(predictions, test_labels,
                                        subject_id= subject_id, dataset_id=dataset_id, save=False)

            evaluate_uncertainty(predictions, test_labels, prediction_proba,
                                 subject_id=subject_id, dataset_id=dataset_id, save=False)

            plot_calibration(predictions, test_labels, prediction_proba,
                             subject_id=subject_id, dataset_id=dataset_id, save=False)


if __name__ == '__main__':
    main()
