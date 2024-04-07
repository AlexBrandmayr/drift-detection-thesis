#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Description:
# This file contains an implementation of a concept drift detector based on the jensen Shannon Divergence
# library: frouros
# reference: https://github.com/IFCA/frouros/blob/main/frouros/detectors/data_drift/batch/distance_based/js.py
# -----------------------------------------------------------------------------------------------------------
from frouros.detectors.data_drift.batch.distance_based.js import JS


class JsConceptDriftDetector:
    ''' Concept drift detector based on the Jensen Shannon divergence using a sliding window approach.

    Attributes:
    -----------
    batch_size : int
        The size of the sliding window used for detecting concept drifts.
    dist_threshold : float
        The threshold value for detecting concept drifts. If the Jensen Shannon divergence distance between the
        current window of data and the reference window is greater than this value, a drift is detected.
    reference_data : array-like or None
        The reference window of data to use for detecting concept drifts. If not provided, the first window of
        data encountered will be used as the reference.
    drift_ind : list
        A list of indices where concept drifts were detected.
    detector : object
        The Jensen Shannon distance object from the frouros.unsupervised.distance_based module used to calculate
        the distance between windows of data.
    cnt_drift : int
        The count of concept drifts detected.
    result_list : list
        A list of the distance values calculated for each window of data tested for drift.

    Methods:
    --------
    detect_drift(new_data):
        Detects concept drift using the Jensen Shannon divergence between the current window of data and the
        reference window.
    test_stat(new_data):
        Calculates the Jensen Shannon divergence distance between the current window of data and the reference
        window.
    detect_drift_window(data_stream, overlapping=False):
        Detects concept drifts in a sliding window approach using the `detect_drift` method on consecutive windows
        of data. If `overlapping` is True, the sliding window will overlap between consecutive windows.
    '''

    def __init__(self, batch_size, threshold):
        self.batch_size = batch_size
        self.threshold = threshold
        self.reference_data = None
        self.drift_ind = []
        self.detector = JS()
        self.cnt_drift = 0
        self.result_list = []
        self.distance = None

    def detect_drift(self, new_data):
        if self.reference_data is None:
            self.reference_data = new_data

        self.detector.fit(self.reference_data)

        result = self.detector.compare(new_data)[0]
        self.distance = result[0]

        if result[0] > self.threshold:
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
