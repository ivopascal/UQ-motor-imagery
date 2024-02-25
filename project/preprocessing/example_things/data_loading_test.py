import mne
import pandas as pd
from moabb.datasets import BNCI2014_001

dataset = BNCI2014_001()

# Load the data for a specific subject
subject_data = dataset.get_data(subjects=[1])

# This is when you want to process data from the first session and the first run
session_name = list(subject_data[1].keys())[0]
run_name = list(subject_data[1][session_name].keys())[0]
raw_data = subject_data[1][session_name][run_name].load_data()

# Accessing the raw data and events
data, times = raw_data[:, :]  # data: numpy array of shape (n_channels, n_times)
events, event_ids = mne.events_from_annotations(raw_data)

# Convert the data to a pandas DataFrame
df = pd.DataFrame(data.T, columns=raw_data.ch_names)

# Now `df` is a pandas DataFrame with each column representing a channel and rows represent timepoints in th eeg
print(df)