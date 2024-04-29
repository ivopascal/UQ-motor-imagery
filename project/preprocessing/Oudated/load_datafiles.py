import pandas as pd
import os
import numpy as np


def construct_filename(subject_id, trial_id, base_dir="../../data/", prefix="", datatype=".csv"):
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
    filename = f"{prefix}subject_{subject_id}_trial_{trial_id}{datatype}"
    full_path = os.path.join(base_dir, filename)
    return full_path


def read_data_me(subject_id, trial_id, base_dir="../../data/data_mytry/preprocessed"):
    """
    Reads the data file corresponding to the given subject and trial numbers into a pandas DataFrame.

    Parameters:
    - subject_id: int, the subject number.
    - trial_id: int, the trial number.

    Returns:
    - pandas.DataFrame: the data loaded from the file.
    """
    # Construct the filename based on the subject and trial numbers
    file_path = construct_filename(subject_id, trial_id, base_dir)

    # Check if the file exists
    if not os.path.isfile(file_path):
        print("File not found")
        raise FileNotFoundError(f"No data file found at {file_path}")

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    return df


def read_data_moabb(subject_id, trial_id, base_dir="../../data/data_moabb_try/preprocessed"):

    # Replace 'path_to_file' with the actual file paths you saved the arrays to
    x_file_path = construct_filename(subject_id, trial_id, base_dir, "X_preprocessed_", ".npy")
    y_file_path = construct_filename(subject_id, trial_id, base_dir, "y_preprocessed_", ".npy")
    metadata_file_path = construct_filename(subject_id, trial_id, base_dir, "metadata_preprocessed_", ".npy")

    # Loading the arrays from the saved files
    X_loaded = np.load(x_file_path, allow_pickle=True)
    y_loaded = np.load(y_file_path, allow_pickle=True)
    metadata_loaded = np.load(metadata_file_path, allow_pickle=True)

    return X_loaded, y_loaded, metadata_loaded


# def main():
#     x, y, metadata = read_data_moabb(1, 1)
#     print(x,y,metadata)
#
#
# if __name__ == '__main__':
#     main()