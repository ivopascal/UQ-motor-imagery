import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mne import concatenate_raws, pick_types, events_from_annotations, Epochs
from moabb.datasets import BNCI2014_001
from moabb.evaluations import CrossSessionEvaluation, WithinSessionEvaluation
from moabb.paradigms import MotorImagery
from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from sklearn.model_selection import KFold, cross_val_score
import seaborn as sns
from sklearn.pipeline import make_pipeline

from Thesis.project.preprocessing.load_datafiles import read_data_moabb
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)


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


def main():
    # base_dir = "../../../../data/data_moabb_try/preprocessed"
    # X, y, groups = load_all_data(base_dir)

    dataset = BNCI2014_001()
    paradigm = MotorImagery(
        n_classes=4, fmin=7.5, fmax=30, tmin=0, tmax=None
    )

    subjects = [1,2,3,4,5,6,7,8]  # You can add more subjects
    dataset.subject_list = subjects
    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=subjects)

    mdm = MDM(metric=dict(mean='riemann', distance='riemann'))

    pipeline = make_pipeline(Covariances(estimator='lwf'), MDM())

    cv = KFold(n_splits=8, shuffle=True, random_state=42)

    # Evaluate the pipeline using cross-validation
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
    print(f"Accuracy: {np.mean(scores):.2f} ± {np.std(scores):.2f}")

    evaluation = WithinSessionEvaluation(
        paradigm=paradigm,
        datasets=[dataset],
        overwrite=True,
        hdf5_path=None,
    )

    results = evaluation.process({"mdm": pipeline})

    print("Results: ", results)

    fig, ax = plt.subplots(figsize=(8, 7))
    results["subj"] = results["subject"].apply(str)
    sns.barplot(
        x="score", y="subj", hue="session", data=results, orient="h", palette="viridis", ax=ax
    )
    plt.show()


    # Printing the results
    class_balance = np.mean(y == y[0])
    class_balance = max(class_balance, 1. - class_balance)
    print("MDM Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
                                                                  class_balance))


if __name__ == '__main__':
    main()
