#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Description:
# This file contains an implementation of the Kolmogorov Smirnov Drift Detection algorithm based on scipy.stats
# Kolmogorov Smirnov Test
# Library: scipy
#  Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html
# -----------------------------------------------------------------------------------------------------------
from scipy.stats import ks_2samp


class KS_Concept_Drift_Detector:
    """
    Concept Drift Detector based on the Kolmogorov-Smirnov Test using a sliding window approach.

    Attributes:
        batch_size (int): Size of the data batches used for drift detection.
        significance_level (float): The significance level for drift detection.
        reference_data (array-like): Reference data used for drift detection.
        drift_ind (list): List to store indices where concept drift is detected.
        cnt_drift (int): Counter to keep track of the number of detected concept drifts.
        result_list (list): List to store p-values of drift detection results.

    Methods:
        __init__: Initializes the KS_Concept_Drift_Detector with specified parameters.
        detect_drift: Detects concept drift in a given batch of new data.
        test_stat: Computes the Kolmogorov-Smirnov test statistic and p-value for a given batch of data.
        detect_drift_window: Monitors a data stream for concept drifts using batches of data.

    Reference:
        - Library: scipy
        - Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html
    """

    def __init__(self, batch_size, significance_level):
        """
        Initializes the KS_Concept_Drift_Detector with specified parameters.

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
        self.cnt_drift = 0
        self.result_list = []

    def detect_drift(self, new_data):
        """
        Detects concept drift in a given batch of new data using the Kolmogorov-Smirnov Test.

        Args:
            new_data (array-like): The new data batch to analyze for concept drift.

        Returns:
            bool: True if concept drift is detected, False otherwise.
        """

        if self.reference_data is None:
            self.reference_data = new_data

        ks_stat, p_value = ks_2samp(new_data, self.reference_data)

        if p_value < self.significance_level:
            return True
        else:
            return False

    def test_stat(self, new_data):
        """
        Computes the Kolmogorov-Smirnov test statistic and p-value for a given batch of data.

        Args:
            new_data (array-like): The data batch to analyze for concept drift.

        Returns:
            tuple: A tuple containing the Kolmogorov-Smirnov test statistic and p-value.
        """
        if self.reference_data is None:
            self.reference_data = new_data

        return ks_2samp(new_data, self.reference_data)

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
                    self.cnt_drift += 1
                    self.drift_ind.append(i + self.batch_size - 1)
                    ks_stat, p_value = self.test_stat(batch_data)
                    self.result_list.append(p_value)
                    self.reference_data = batch_data

        else:
            for i in range(0, data_stream.shape[0], self.batch_size):
                batch_data = data_stream[i:i + self.batch_size]
                if self.detect_drift(batch_data):
                    print(f'Concept drift detected at index {i + self.batch_size - 1}')
                    self.cnt_drift += 1
                    self.drift_ind.append(i)
                    ks_stat, p_value = self.test_stat(batch_data)
                    self.result_list.append(p_value)
                    self.reference_data = batch_data

        return {'drift_ind': self.drift_ind, 'result_list': self.result_list, 'cnt_drift': self.cnt_drift}
