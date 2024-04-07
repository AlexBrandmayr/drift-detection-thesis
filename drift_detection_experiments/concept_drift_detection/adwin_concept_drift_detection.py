#!/usr/bin/env python#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Description:
# This file contains an implementation of the Adaptive Windowing concept drift detector (ADWIN)
# library: river
# reference: https://riverml.xyz/dev/api/drift/ADWIN/
# -------------------------------------------------------------------------------------------------------
from river.drift import ADWIN


class AdwinConceptDriftDetector:
    """
    Implementation of the Adaptive Windowing (ADWIN) concept drift detector.

    Attributes:
        adwin: ADWIN object from the river library initialized with specified parameters.
        drift_ind (list): List to store indices where concept drift is detected.
        cnt_drift (int): Counter to keep track of the number of detected concept drifts.
        result_list (list): List to store ADWIN estimations at the time of drift detection.

    Methods:
        __init__: Initializes the AdwinConceptDriftDetector with specified parameters.
        detect_drift_window: Detects concept drifts in the given data stream.

    Reference:
    - Library: river
    - Reference: https://riverml.xyz/dev/api/drift/ADWIN/
    """

    def __init__(self, significance_level, clock, min_window_length, grace_period):
        """
        Initializes the AdwinConceptDriftDetector with specified parameters.

        Args:
            significance_level (float): The statistical significance level for drift detection.
            clock (int): The maximum number of elements stored in the window.
            min_window_length (int): The minimum number of instances that must be observed before drift detection begins
            grace_period (int): The number of instances to observe before starting to detect drifts.

        Returns:
            None
        """

        self.adwin = ADWIN(delta=significance_level, clock=clock, min_window_length=min_window_length,
                           grace_period=grace_period)
        self.drift_ind = []
        self.cnt_drift = 0
        self.result_list = []

    def detect_drift_window(self, data_stream):
        """
        Detects concept drifts in the given data stream.

        Args:
            data_stream (iterable): The input data stream to monitor for concept drifts.

        Returns:
            dict: A dictionary containing the following information:
                - 'drift_ind' (list): Indices where concept drift is detected.
                - 'result_list' (list): ADWIN estimations at the time of drift detection.
                - 'cnt_drift' (int): Number of detected concept drifts.
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
