#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Description:
# This file contains an implementation of the ADWIN concept drift detector based on the python river library
# -------------------------------------------------------------------------------------------------------
from river.drift import ADWIN


class AdwinConceptDriftDetector:
    """ ADWIN Concept drift detector

     Attributes:
         self.adwin
         self.drift_ind (list)   list of indices of detected drifts
         self.cnt_drift (int)    number of detected drifts
         self.result_list (list)  list of estimated means

    Methods
         detect_drift_window       performs the drift detection on datastream
         """

    def __init__(self, significance_level,clock, min_window_length,grace_period):
        """
        Initialize a new instance of the class.

        Args:
            significance_level (float): significance level  of the ADWIN  concept drift detector.

        Returns:
            None
        """
        self.adwin = ADWIN(delta=significance_level,clock=clock, min_window_length=min_window_length,grace_period=grace_period)
        self.drift_ind = []
        self.cnt_drift = 0
        self.result_list = []

    def detect_drift_window(self, data_stream):
        """
        Initialize a new instance of the class.

        Args:
            data_stream (np.array): Data stream to perform the drift detection on

        Returns:
           dictionary of drift indices , estimation results and counted drifts
        """
        for i, val in enumerate(data_stream):

            _ = self.adwin.update(val)

            if self.adwin.drift_detected:
                print(f'Concept drift detected at index {i}')
                self.cnt_drift += 1
                self.drift_ind.append(i)

                self.result_list.append(self.adwin.estimation)
                self.adwin._reset()

        return {'drift_ind': self.drift_ind, 'result_list': self.result_list, 'cnt_drift': self.cnt_drift}