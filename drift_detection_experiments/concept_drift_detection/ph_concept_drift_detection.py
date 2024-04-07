#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Description:
# Univariate Concept drift detection based on the Page Hinkley Test
# library: river
# reference: https://riverml.xyz/dev/api/drift/PageHinkley/
# -----------------------------------------------------------------------------------------------------------
from river.drift import PageHinkley


class PageHinkleyConceptDriftDetector:
    """
    Concept Drift Detector based on the Page Hinkley Test.

    Attributes:
        ph: Instance of PageHinkley for detecting concept drift.
        drift_ind (list): List to store indices where concept drift is detected.
        cnt_drift (int): Counter to keep track of the number of detected concept drifts.
        result_list (list): List to store detection results.

    Methods:
        __init__: Initializes the PageHinkleyConeptDriftDetector with specified parameters.
        detect_drift_window: Monitors a data stream for concept drifts.

    Reference:
        - Library: river
        - Reference: https://riverml.xyz/dev/api/drift/PageHinkley/
    """

    def __init__(self, min_instances, delta, threshold):
        """
        Initializes the PageHinkleyConeptDriftDetector with specified parameters.

        Args:
            min_instances (int): The minimum number of instances before drift can be detected.
            delta (float): The delta parameter controls the sensitivity to drift.
            threshold (float): The drift detection threshold.

        Returns:
            None
        """
        self.ph = PageHinkley(min_instances, delta, threshold)
        self.drift_ind = []
        self.cnt_drift = 0
        self.result_list = []

    def detect_drift_window(self, data_stream):
        """
        Monitors a data stream for concept drifts using the Page Hinkley Test.

        Args:
            data_stream (array-like): The data stream to monitor for concept drifts.

        Returns:
            dict: A dictionary containing the following information:
                - 'drift_ind' (list): Indices where concept drift is detected.
                - 'result_list' (list): List of detection results.
                - 'cnt_drift' (int): Number of detected concept drifts.
        """
        for i, val in enumerate(data_stream):

            _ = self.ph.update(val)

            if self.ph.drift_detected:
                print(f'Concept drift detected at index {i}')
                self.cnt_drift += 1
                self.drift_ind.append(i)
                self.ph._reset()

                self.result_list.append(0)

        return {'drift_ind': self.drift_ind, 'result_list': self.result_list, 'cnt_drift': self.cnt_drift}
