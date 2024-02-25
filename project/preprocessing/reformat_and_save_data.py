import os

import pandas as pd
from moabb.datasets import BNCI2014_001

# Initialize the dataset
dataset = BNCI2014_001()

# Define the number of subjects and trials
num_subjects = 9
num_trials_per_subject = 2

# Loop over each subject
for subject_id in range(1, num_subjects + 1):
    # Get the data for the current subject
    subject_data = dataset.get_data(subjects=[subject_id])

    # Assuming the structure of subject_data allows direct access to sessions and runs
    # Loop over each trial (session/run)
    for trial_id in range(1, num_trials_per_subject + 1):
        # For simplicity, this assumes sessions and runs can be directly indexed
        # You might need to adjust this based on the actual data structure
        session_name = list(subject_data[subject_id].keys())[trial_id - 1]  # Adjust based on actual structure
        run_name = list(subject_data[subject_id][session_name].keys())[0]  # Assuming first run if multiple
        raw_data = subject_data[subject_id][session_name][run_name].load_data()

        # Convert the raw data to a DataFrame
        data, times = raw_data[:, :]
        df = pd.DataFrame(data.T, columns=raw_data.ch_names)

        # Write the DataFrame to a CSV file
        filename = f"subject_{subject_id}_trial_{trial_id}.csv"
        full_path = os.path.join("../../data/rawData", filename)

        # Write the DataFrame to a CSV file at the specified path
        df.to_csv(full_path, index=False)
        print(f"Data for Subject {subject_id}, Trial {trial_id} written to {full_path}")
