#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Description:
# This file contains a univariate concept drift detector based on the Cramer von Mises Test
# library: frouros
# reference: https://github.com/IFCA/frouros/blob/main/frouros/detectors/data_drift/batch/statistical_test/cvm.py
# -----------------------------------------------------------------------------------------------------------
from frouros.detectors.data_drift.batch.statistical_test import cvm


class CvmConceptDriftDetector:
    """
    Concept Drift Detector based on the Cramer von Mises Test.

    Attributes:
        batch_size (int): Size of the data batches used for drift detection.
        significance_level (float): The significance level for drift detection.
        reference_data (array-like): Reference data used for drift detection.
        drift_ind (list): List to store indices where concept drift is detected.
        detector: Instance of CVMTest for conducting the Cramer von Mises Test.
        cnt_drift (int): Counter to keep track of the number of detected concept drifts.
        result_list (list): List to store p-values of drift detection results.
        p_value (float): The p-value of the most recent drift detection.

    Methods:
        __init__: Initializes the CvmConceptDriftDetector with specified parameters.
        detect_drift: Detects concept drift in a given batch of new data.
        detect_drift_window: Monitors a data stream for concept drifts.

    Reference:
        - Library: frouros
        - Reference: https://github.com/IFCA/frouros/blob/main/frouros/detectors/data_drift/batch/statistical_test/cvm.py
    """

    def __init__(self, batch_size, significance_level):
        """
        Initializes the CvmConceptDriftDetector with specified parameters.

        Args:
            batch_size (int): Size of the data batches used for drift detection.
            significance_level (float): The significance level for drift detection.

        Returns:
            None
        """
        self.batch_size = batch_size
        self.significance_level = significance_level
        self.reference_data = None
        self.drift_ind = []
        self.detector = cvm.CVMTest()
        self.cnt_drift = 0
        self.result_list = []
        self.p_value = None

    def detect_drift(self, new_data):
        """
        Detects concept drift in a given batch of new data using the Cramer von Mises Test.

        Args:
            new_data (array-like): The new data batch to analyze for concept drift.

        Returns:
            bool: True if concept drift is detected, False otherwise.
        """
        if self.reference_data is None:
            self.reference_data = new_data

        self.detector.fit(self.reference_data)

        result = self.detector.compare(new_data)[0]
        self.p_value = result.p_value

        if result.p_value < self.significance_level:
            return True
        else:
            return False

    def detect_drift_window(self, data_stream, overlapping=False):
        """
        Monitors a data stream for concept drifts using batches of data.

        Args:
            data_stream (array-like): The data stream to monitor for concept drifts.
            overlapping (bool, optional): If True, allow overlapping batches. Default is False.

        Returns:
            dict: A dictionary containing the following information:
                - 'drift_ind' (list): Indices where concept drift is detected.
                - 'result_list' (list): List of p-values from drift detection results.
                - 'cnt_drift' (int): Number of detected concept drifts.
        """

        if overlapping:

            for i in range(len(data_stream) - self.batch_size + 1):
                batch_data = data_stream[i:i + self.batch_size]
                if self.detect_drift(batch_data):
                    print(f'Concept drift detected at index {i + self.batch_size - 1}')
                    self.drift_ind.append(i + self.batch_size - 1)
                    self.cnt_drift += 1
                    self.result_list.append(self.p_value)
                    self.reference_data = batch_data

        else:

            for i in range(0, data_stream.shape[0], self.batch_size):
                batch_data = data_stream[i:i + self.batch_size]
                if self.detect_drift(batch_data):
                    print(f'Concept drift detected at index {i + self.batch_size - 1}')
                    self.drift_ind.append(i)
                    self.cnt_drift += 1
                    self.result_list.append(self.p_value)
                    self.reference_data = batch_data

        return {'drift_ind': self.drift_ind, 'result_list': self.result_list, 'cnt_drift': self.cnt_drift}
