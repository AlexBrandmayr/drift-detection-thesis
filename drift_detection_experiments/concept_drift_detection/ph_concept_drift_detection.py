#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Description:
# Univariate Concept drift detection based on the Page Hinkley Test
# library: river
# reference: https://riverml.xyz/dev/api/drift/PageHinkley/
# -----------------------------------------------------------------------------------------------------------
from river.drift import PageHinkley


class PageHinkleyConeptDriftDetector:
    ''' Concept drift detector based on the Page Hinkley Test using a sliding window approach.

    Attributes:
    -----------
    window_size : int
        The size of the sliding window used for detecting concept drifts.
    dist_threshold : float
        The threshold value for detecting concept drifts. If the average Population Stability Index between the
        current window of data and the reference window is greater than this value, a drift is detected.
    num_bins: int
        The  number of bins used for creating the histogram
    reference_data : array-like or None
        The reference wind data to use for detecting concept drifts.
    drift_ind : list
        A list of indices where concept drifts were detected.
    cnt_drift : int
        The count of concept drifts detected.
    result_list : list
        A list of the distance values calculated for each window of data tested for drift.

    Methods:
    --------
    detect_drift(new_data):
        Detects concept drift by computing the average Population Stability Index between the current window of data and the
        reference window.
    test_stat(new_data):
        Calculates the average Population Stability Index between the current window of data and the reference
        window.
    detect_drift_window(data_stream, overlapping=False):
        Detects concept drifts in a sliding window approach using the `detect_drift` method on consecutive windows
        of data. If `overlapping` is True, the sliding window will overlap between consecutive windows.
    '''

    def __init__(self, min_instances, delta, threshold):
        self.ph = PageHinkley(min_instances, delta, threshold)
        self.drift_ind = []
        self.cnt_drift = 0
        self.result_list = []

    def detect_drift_window(self, data_stream):
        for i, val in enumerate(data_stream):

            _ = self.ph.update(val)

            if self.ph.drift_detected:
                print(f'Concept drift detected at index {i}')
                self.cnt_drift += 1
                self.drift_ind.append(i)
                self.ph._reset()

                self.result_list.append(0)

        return {'drift_ind': self.drift_ind, 'result_list': self.result_list, 'cnt_drift': self.cnt_drift}
