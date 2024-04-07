#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Description:
# This file contains an implementation of the Kolmogorov Smirnov Drift Detection algorithm based on scipy.stats
# Kolmogorov Smirnov Test
# -----------------------------------------------------------------------------------------------------------
from scipy.stats import ks_2samp


class KS_Concept_Drift_Detector:
    """Concept Drift Detector based on the Kolmogorov Smirnov Test using a sliding window approach."""

    def __init__(self, batch_size, significance_level):
        self.batch_size = batch_size
        self.significance_level = significance_level
        self.reference_data = None
        self.drift_ind = []
        self.cnt_drift = 0
        self.result_list = []

    def detect_drift(self, new_data):
        if self.reference_data is None:
            self.reference_data = new_data

        ks_stat, p_value = ks_2samp(new_data, self.reference_data)

        if p_value < self.significance_level:
            return True
        else:
            return False

    def test_stat(self, new_data):
        if self.reference_data is None:
            self.reference_data = new_data

        return ks_2samp(new_data, self.reference_data)

    def detect_drift_window(self, data_stream, overlapping=False):

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
