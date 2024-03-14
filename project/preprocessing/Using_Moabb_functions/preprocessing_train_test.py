import os

import numpy as np
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

from Thesis.project.preprocessing.load_datafiles_traintest import construct_filename


def saveFiles(X, y, train_test='train', base_dir="../../../data/train_test_data"):

    preprocess_dir = os.path.join(base_dir, "preprocessed")
    if not os.path.exists(preprocess_dir):
        os.makedirs(preprocess_dir)

    # Construct filenames for X, y, and metadata
    x_file = construct_filename(train_test, preprocess_dir, "X_preprocessed_", ".npy")
    y_file = construct_filename(train_test, preprocess_dir, "y_preprocessed_", ".npy")

    # Save the arrays
    np.save(x_file, X)
    np.save(y_file, y)

    print(f"Preprocessed data written for {train_test}")
    print(f"Written to {preprocess_dir}")


def main():
    dataset = BNCI2014_001()
    paradigm = MotorImagery(
        n_classes=4, fmin=7.5, fmax=30, tmin=0, tmax=None
    )

    num_subjects = 9
    num_trials_per_subject = 2

    train_subjects = 8

    X_train, y_train, X_test, y_test = [], [], [], []


    for subject_id in range(1, train_subjects + 1):
        for trial_id in range(1, num_trials_per_subject + 1):
            subjects = [subject_id]
            X, y, metadata = paradigm.get_data(dataset=dataset, subjects=subjects)
            # print(X.shape)  #shape is (576, 22, 1001)
            # print(y.shape)  #shape is (576,)
            # print(metadata.shape)   #shape is (576, 3)

            X_train.append(X)
            y_train.append(y)

    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    saveFiles(X_train, y_train, 'train')

    for subject_id in range(train_subjects + 1, num_subjects + 1):
        for trial_id in range(1, num_trials_per_subject + 1):
            subjects = [subject_id]
            X, y, metadata = paradigm.get_data(dataset=dataset, subjects=subjects)
            # print(X.shape)  #shape is (576, 22, 1001)
            # print(y.shape)  #shape is (576,)
            # print(metadata.shape)   #shape is (576, 3)

            X_test.append(X)
            y_test.append(y)

    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    saveFiles(X_test, y_test, 'test')



if __name__ == '__main__':
    main()