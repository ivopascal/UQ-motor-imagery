import os
import mne
import numpy as np
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from pyriemann.estimation import Covariances
from sklearn.pipeline import make_pipeline

from Thesis.project.models.Riemann.MDM_model import MDM
from Thesis.project.preprocessing.load_datafiles import construct_filename



def main():
    dataset = BNCI2014_001()

    dataset.subject_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    sessions = dataset.get_data(subjects=[1, 2, 3, 4, 5, 6, 7, 8, 9])

    #print(sessions)

    num_subjects = 9

    for subject_id in range(1, num_subjects + 1):
        session_name = "0train"
        run_name = "0"
        raw = sessions[subject_id][session_name][run_name]

        #print(raw)

        data, times = raw[:, :]  # data: numpy array of shape (n_channels, n_times)
        events, event_ids = mne.events_from_annotations(raw)

        print("Data: ", data)
        print(data.shape)
        #Data:  [[ 3.41796875e-07 -6.34765625e-06 -1.80664062e-06 ...  2.92968750e-07
             #  -6.34765625e-06 -9.71679687e-06]  ...  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00 ...  0.00000000e+00
             #   0.00000000e+00  0.00000000e+00]]
        #Shape = (26, 96735)

        #print("Times: ", times)
        #print(times.shape)
        # Times: [0.00000e+00 4.00000e-03 8.00000e-03... 3.86928e+02 3.86932e+02
        #         3.86936e+02]
        # Shape = (96735,)

        print("Events: ", events)
        print(events.shape)
        # [[  250     0     4] ... [94757     0     2]]
        # Shape = (48, 3)

        print("Event_Ids: ", event_ids)
        # Event_Ids:  {'feet': 1, 'left_hand': 2, 'right_hand': 3, 'tongue': 4}



if __name__ == '__main__':
    main()