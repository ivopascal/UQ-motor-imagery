import pandas as pd
from scipy.signal import butter, sosfilt
from Thesis.project.preprocessing.load_datafiles import construct_filename, read_data


def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    Designs a Butterworth bandpass filter.

    Parameters:
    - lowcut: float, the low cutoff frequency.
    - highcut: float, the high cutoff frequency.
    - fs: float, the sampling rate.
    - order: int, the order of the filter.

    Returns:
    - sos: array, the second-order sections of the bandpass filter.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def apply_bandpass_filter(data, lowcut=7.5, highcut=30.0, fs=250.0, order=5):
    """
    Applies a bandpass filter to the data.

    Parameters:
    - data: DataFrame, the input data.
    - lowcut: float, the low cutoff frequency.
    - highcut: float, the high cutoff frequency.
    - fs: float, the sampling rate.
    - order: int, the order of the filter.

    Returns:
    - DataFrame: the filtered data.
    """
    sos = butter_bandpass(lowcut, highcut, fs, order)
    filtered_data = sosfilt(sos, data, axis=0)
    return pd.DataFrame(filtered_data, columns=data.columns)


def process_files(subject_id, trial_id):
    """
    Processes all subject data files by applying a bandpass filter and saving the results.

    Parameters:
    - num_subjects: int, the number of subjects.
    - num_trials_per_subject: int, the number of trials per subject.
    - base_dir: str, the base directory where the data files are stored.
    """

    input_file = read_data(subject_id, trial_id)
    preprocessed_df = apply_bandpass_filter(input_file)
    return preprocessed_df


