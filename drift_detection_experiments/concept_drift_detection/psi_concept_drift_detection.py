#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Description:
#  This file contains a univariate concept drift detector based on the population stability index
# library: numpy /pandas
# reference: https://medium.com/model-monitoring-psi/population-stability-index-psi-ab133b0a5d42
# -----------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd


class PsiConceptDriftDetector:
    ''' Concept drift detector based on the Population Stability Index using a sliding window approach.

    Attributes:
    -----------
    batch_size : int
        The size of the sliding window used for detecting concept drifts.
    threshold : float
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
        Detects concept drift by computing the average Population Stability Index between the current window of data and
         the reference window.
    test_stat(new_data):
        Calculates the average Population Stability Index between the current window of data and the reference
        window.
    detect_drift_window(data_stream, overlapping=False):
        Detects concept drifts in a sliding window approach using the `detect_drift` method on consecutive windows
        of data. If `overlapping` is True, the sliding window will overlap between consecutive windows.
    '''
    def __init__(self, batch_size, threshold, num_bins):
        self.batch_size = batch_size
        self.threshold = threshold
        self.reference_data = None
        self.drift_ind = []
        self.num_bins = num_bins
        self.cnt_drift = 0
        self.result_list = []
        self.distance = None

    def detect_drift(self, new_data):
        if self.reference_data is None:
            self.reference_data = new_data

        psi_result = np.mean(self._psi(self.reference_data, new_data, self.num_bins))
        self.distance = psi_result

        if psi_result > self.threshold:
            return True
        else:
            return False

    def _psi(self, score_initial, score_new, num_bins=10, mode='fixed'):
        eps = 1e-4

        # Sort the data
        score_initial.sort()
        score_new.sort()

        # Prepare the bins
        min_val = min(min(score_initial), min(score_new))
        max_val = max(max(score_initial), max(score_new))
        if mode == 'fixed':
            bins = [min_val + (max_val - min_val) * i / num_bins for i in range(num_bins + 1)]
        elif mode == 'quantile':
            bins = pd.qcut(score_initial, q=num_bins, retbins=True)[
                1]  # Create the quantiles based on the initial population
        else:
            raise ValueError(f"Mode \'{mode}\' not recognized. Your options are \'fixed\' and \'quantile\'")
        bins[0] = min_val - eps  # Correct the lower boundary
        bins[-1] = max_val + eps  # Correct the higher boundary

        # Bucketize the initial population and count the sample inside each bucket
        bins_initial = pd.cut(score_initial, bins=bins, labels=range(1, num_bins + 1))
        df_initial = pd.DataFrame({'initial': score_initial, 'bin': bins_initial})
        grp_initial = df_initial.groupby('bin').count()
        grp_initial['percent_initial'] = grp_initial['initial'] / sum(grp_initial['initial'])

        # Bucketize the new population and count the sample inside each bucket
        bins_new = pd.cut(score_new, bins=bins, labels=range(1, num_bins + 1))
        df_new = pd.DataFrame({'new': score_new, 'bin': bins_new})
        grp_new = df_new.groupby('bin').count()
        grp_new['percent_new'] = grp_new['new'] / sum(grp_new['new'])

        # Compare the bins to calculate PSI
        psi_df = grp_initial.join(grp_new, on="bin", how="inner")

        # Add a small value for when the percent is zero
        psi_df['percent_initial'] = psi_df['percent_initial'].apply(lambda x: eps if x == 0 else x)
        psi_df['percent_new'] = psi_df['percent_new'].apply(lambda x: eps if x == 0 else x)

        # Calculate the psi
        psi_df['psi'] = (psi_df['percent_initial'] - psi_df['percent_new']) * np.log(
            psi_df['percent_initial'] / psi_df['percent_new'])

        # Return the psi values
        return psi_df['psi'].values

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
