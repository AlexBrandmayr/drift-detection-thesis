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
    """
    Concept Drift Detector based on the Population Stability Index (PSI).

    Attributes:
        batch_size (int): Size of the data batches used for drift detection.
        threshold (float): The threshold value for drift detection.
        reference_data (array-like): Reference data used for drift detection.
        drift_ind (list): List to store indices where concept drift is detected.
        num_bins (int): Number of bins used for computing PSI.
        cnt_drift (int): Counter to keep track of the number of detected concept drifts.
        result_list (list): List to store PSI values of drift detection results.
        distance (float): The PSI value of the most recent drift detection.

    Methods:
        __init__: Initializes the PsiConceptDriftDetector with specified parameters.
        detect_drift: Detects concept drift in a given batch of new data.
        _psi: Computes the Population Stability Index (PSI) between two datasets.
        detect_drift_window: Monitors a data stream for concept drifts using batches of data.

    Reference:
        - Reference: https://medium.com/model-monitoring-psi/population-stability-index-psi-ab133b0a5d42
    """

    def __init__(self, batch_size, threshold, num_bins):
        """
        Initializes the PsiConceptDriftDetector with specified parameters.

        Args:
            batch_size (int): Size of the data batches used for drift detection.
            threshold (float): The threshold value for drift detection.
            num_bins (int): Number of bins used for computing PSI.

        Returns:
            None
        """
        self.batch_size = batch_size
        self.threshold = threshold
        self.reference_data = None
        self.drift_ind = []
        self.num_bins = num_bins
        self.cnt_drift = 0
        self.result_list = []
        self.distance = None

    def detect_drift(self, new_data):
        """
         Detects concept drift in a given batch of new data using the Population Stability Index (PSI).

         Args:
             new_data (array-like): The new data batch to analyze for concept drift.

         Returns:
             bool: True if concept drift is detected, False otherwise.
         """
        if self.reference_data is None:
            self.reference_data = new_data

        psi_result = np.mean(self._psi(self.reference_data, new_data, self.num_bins))
        self.distance = psi_result

        if psi_result > self.threshold:
            return True
        else:
            return False

    def _psi(self, score_initial, score_new, num_bins=10, mode='fixed'):
        """
        Computes the Population Stability Index (PSI) between two datasets.

        Args:
            score_initial (array-like): Initial data batch.
            score_new (array-like): New data batch.
            num_bins (int, optional): Number of bins used for computing PSI. Default is 10.
            mode (str, optional): Mode for binning method ('fixed' or 'quantile'). Default is 'fixed'.

        Returns:
            array-like: PSI values between the two datasets.
        """
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
        """
        Monitors a data stream for concept drifts using batches of data.

        Args:
            data_stream (array-like): The data stream to monitor for concept drifts.
            overlapping (bool, optional): If True, allow overlapping batches. Default is False.

        Returns:
            dict: A dictionary containing the following information:
                - 'drift_ind' (list): Indices where concept drift is detected.
                - 'result_list' (list): List of PSI values from drift detection results.
                - 'cnt_drift' (int): Number of detected concept drifts.
        """

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
