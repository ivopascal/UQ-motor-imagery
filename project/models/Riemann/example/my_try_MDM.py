import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mne import concatenate_raws, pick_types, events_from_annotations, Epochs
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from sklearn.model_selection import KFold, cross_val_score
import seaborn as sns

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
    # base_dir = "../../../../data/data_moabb_try/preprocessed"
    # X, y, groups = load_all_data(base_dir)

    dataset = BNCI2014_001()
    paradigm = MotorImagery(
        n_classes=4, fmin=7.5, fmax=30, tmin=0, tmax=None
    )

    subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=subjects, return_raws=True)

    raw = concatenate_raws(X)

    picks = pick_types(
        raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
    # subsample elecs
    picks = picks[::2]

    # Apply band-pass filter
    raw.filter(7.5, 30., method='iir', picks=picks)

    #event_id = dict(feet=1, left_hand=2, right_hand=3, tongue=4)    #dit werkt wel stuk minder goed
    event_id = dict(hands=2, feet=3)

    events, _ = events_from_annotations(raw)    #, event_id=dict(T1=2, T2=3))

    tmin, tmax = 1., 2.
    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    epochs = Epochs(
        raw,
        events,
        event_id,
        tmin,
        tmax,
        proj=True,
        picks=picks,
        baseline=None,
        preload=True,
        verbose=False)
    labels = epochs.events[:, -1] - 2

    # cross validation
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    # get epochs
    epochs_data_train = 1e6 * epochs.get_data(copy=False)

    # compute covariance matrices
    cov_data_train = Covariances().transform(epochs_data_train)

    ###############################################################################
    # Classification with Minimum Distance to Mean
    # --------------------------------------------

    mdm = MDM(metric=dict(mean='riemann', distance='riemann'))

    # Use scikit-learn Pipeline with cross_val_score function
    scores = cross_val_score(mdm, cov_data_train, labels, cv=cv, n_jobs=1)

    # Printing the results
    class_balance = np.mean(labels == labels[0])
    class_balance = max(class_balance, 1. - class_balance)
    print("MDM Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
                                                                  class_balance))

    mdm = MDM()
    mdm.fit(cov_data_train, labels)

    fig, axes = plt.subplots(1, 2, figsize=[8, 4])
    ch_names = [ch.replace('.', '') for ch in epochs.ch_names]

    df = pd.DataFrame(data=mdm.covmeans_[0], index=ch_names, columns=ch_names)
    g = sns.heatmap(
        df, ax=axes[0], square=True, cbar=False, xticklabels=2, yticklabels=2)
    g.set_title('Mean covariance - hands')

    df = pd.DataFrame(data=mdm.covmeans_[1], index=ch_names, columns=ch_names)
    g = sns.heatmap(
        df, ax=axes[1], square=True, cbar=False, xticklabels=2, yticklabels=2)
    plt.xticks(rotation='vertical')
    plt.yticks(rotation='horizontal')
    g.set_title('Mean covariance - feets')

    # dirty fix
    plt.sca(axes[0])
    plt.xticks(rotation='vertical')
    plt.yticks(rotation='horizontal')
    plt.show()


if __name__ == '__main__':
    main()