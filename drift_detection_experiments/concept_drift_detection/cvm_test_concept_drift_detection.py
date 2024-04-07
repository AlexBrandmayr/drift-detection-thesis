#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Description:
# This file contains a univariate concept drift detector based on the Cramer von Mises Test
# library: frouros
# reference: https://github.com/IFCA/frouros/blob/main/frouros/detectors/data_drift/batch/statistical_test/cvm.py
# -----------------------------------------------------------------------------------------------------------
from frouros.detectors.data_drift.batch.statistical_test import cvm


class CvmConceptDriftDetector:
    '''Concept Drift Detector based on the Cramer von Mises Test'''

    def __init__(self, batch_size, significance_level):
        self.batch_size = batch_size
        self.significance_level = significance_level
        self.reference_data = None
        self.drift_ind = []
        self.detector = cvm.CVMTest()
        self.cnt_drift = 0
        self.result_list = []
        self.p_value = None

    def detect_drift(self, new_data):
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
