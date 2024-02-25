import pandas as pd
import os

def construct_filename(subject_id, trial_id, base_dir="../../data/", prefix=""):
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
    filename = f"{prefix}subject_{subject_id}_trial_{trial_id}.csv"
    full_path = os.path.join(base_dir, filename)
    return full_path

def read_data(subject_id, trial_id, base_dir="../../data/preprocessed"):
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
