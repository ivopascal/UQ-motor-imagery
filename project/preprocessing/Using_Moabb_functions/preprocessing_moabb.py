import os

import numpy as np
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

from Thesis.project.preprocessing.load_datafiles import construct_filename


def saveFiles(X, y, metadata, subject_id, trial_id, base_dir="../../../data/data_moabb_try"):

    preprocess_dir = os.path.join(base_dir, "preprocessed")
    if not os.path.exists(preprocess_dir):
        os.makedirs(preprocess_dir)

    # Construct filenames for X, y, and metadata
    x_file = construct_filename(subject_id, trial_id, preprocess_dir, "X_preprocessed_", ".npy")
    y_file = construct_filename(subject_id, trial_id, preprocess_dir, "y_preprocessed_", ".npy")
    metadata_file = construct_filename(subject_id, trial_id, preprocess_dir, "metadata_preprocessed_", ".npy")

    # Save the arrays
    np.save(x_file, X)
    np.save(y_file, y)
    np.save(metadata_file, metadata)

    print(f"Preprocessed data written for {subject_id, trial_id}")
    print(f"Written to {preprocess_dir}")


def main():
    dataset = BNCI2014_001()
    paradigm = MotorImagery(
        n_classes=4, fmin=7.5, fmax=30, tmin=0, tmax=None
    )

    num_subjects = 9
    # num_trials_per_subject = 2

    for subject_id in range(1, num_subjects + 1):
        # for trial_id in range(1, num_trials_per_subject + 1):
        subjects = [subject_id]
        X, y, metadata = paradigm.get_data(dataset=dataset, subjects=subjects)
        # print(X.shape)  #shape is (576, 22, 1001)
        # print(y.shape)  #shape is (576,)
        # print(metadata.shape)   #shape is (576, 3)

        # HIER GAAT IETS FOUT MET TRIALS, waarom doe ik daar nu niks mee?

        saveFiles(X, y, metadata, subject_id)   #, trial_id)



if __name__ == '__main__':
    main()