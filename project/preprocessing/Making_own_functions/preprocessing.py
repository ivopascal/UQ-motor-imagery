import os

import filter_data
from Thesis.project.preprocessing.load_datafiles import construct_filename


def saveFiles(dataFrame, subject_id, trial_id, base_dir="../../data"):
    preprocess_dir = os.path.join(base_dir, "preprocessed")
    if not os.path.exists(preprocess_dir):
        os.makedirs(preprocess_dir)
    output_file = construct_filename(subject_id, trial_id, preprocess_dir, "preprocessed_")

    dataFrame.to_csv(output_file, index=False)
    print(f"Preprocessed data written to {output_file}")


def main():
    num_subjects = 9
    num_trials_per_subject = 2

    for subject_id in range(1, num_subjects + 1):
        for trial_id in range(1, num_trials_per_subject + 1):
            preprocessed_data = filter_data.process_files(subject_id, trial_id)

            saveFiles(preprocessed_data, subject_id, trial_id)



if __name__ == '__main__':
    main()
