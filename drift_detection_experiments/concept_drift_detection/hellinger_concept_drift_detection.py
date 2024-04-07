#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Description:
# This python file conatains a univarate concept drift detector based on the hellinger distance
# library: frouros
# reference :https://github.com/IFCA/frouros/blob/main/frouros/detectors/
# data_drift/batch/distance_based/hellinger_distance.py
# -----------------------------------------------------------------------------------------------------------
from frouros.detectors.data_drift.batch.distance_based import HellingerDistance


class HellingerDistanceDriftDetector:

    def __init__(self, batch_size, threshold):
        self.batch_size = batch_size
        self.threshold = threshold
        self.reference_data = None
        self.drift_ind = []
        self.detector = HellingerDistance()
        self.cnt_drift = 0
        self.result_list = []
        self.distance = None

    def detect_drift(self, new_data):
        if self.reference_data is None:
            self.reference_data = new_data

        self.detector.fit(self.reference_data)

        result = self.detector.compare(new_data)
        self.distance = result[0].distance

        if result[0].distance > self.threshold:
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
                    self.result_list.append(self.distance)
                    self.reference_data = batch_data

        else:

            for i in range(0, data_stream.shape[0], self.batch_size):
                batch_data = data_stream[i:i + self.batch_size]
                if self.detect_drift(batch_data):
                    print(f'Concept drift detected at index {i + self.batch_size - 1}')
                    self.drift_ind.append(i)
                    self.cnt_drift += 1
                    self.result_list.append(self.distance)
                    self.reference_data = batch_data

        return {'drift_ind': self.drift_ind, 'result_list': self.result_list, 'cnt_drift': self.cnt_drift}