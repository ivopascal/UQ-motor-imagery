import os

import numpy as np


def construct_filename(train_test='train', base_dir="../../data/", prefix="", datatype=".csv"):
    """
    Constructs the filename based on the subject and trial numbers.

    Parameters:
    - subject_id: int, the subject number.
    - trial_id: int, the trial number.
    - base_dir: str, the base directory where the data files are stored.
    - prefix: str, optional prefix for the filename, e.g., for preprocessed files.

    Returns:
    - str: the full path to the data file.
    """
    filename = f"{prefix}{train_test}{datatype}"
    full_path = os.path.join(base_dir, filename)
    return full_path


def read_data_traintest(train_test='train', base_dir="../../data/train_test_data/preprocessed"):

    # Replace 'path_to_file' with the actual file paths you saved the arrays to
    x_file_path = construct_filename(train_test, base_dir, "X_preprocessed_", ".npy")
    y_file_path = construct_filename(train_test, base_dir, "y_preprocessed_", ".npy")

    # Loading the arrays from the saved files
    X_loaded = np.load(x_file_path, allow_pickle=True)
    y_loaded = np.load(y_file_path, allow_pickle=True)

    return X_loaded, y_loaded