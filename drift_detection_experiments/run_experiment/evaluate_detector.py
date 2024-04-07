#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Description:
# File to run a series of drift detection experiments to compute the number of true positive, false positive and false
# negatives  on a dataframe and save the results
# -----------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import time
import json
import importlib
import sys


# Load the labels:
label_path = 'labels_df'
label_df = pd.read_pickle(label_path)

with open('series_config.json') as f:
    config = json.load(f)

sys.path.append(".../drift-detection-thesis/drift_detection_experiments/concept_drift_detection")

tag_list = config['drift_detection']['tag_list']
df_name = config['drift_detection']['data_frame']

detector_module, detector_class = config['detector']['class'].rsplit('.', 1)
DetectorClass = getattr(importlib.import_module(detector_module), detector_class)

detector_params = config['detector']['params']

drift_df = pd.read_pickle(df_name)
time_total = 0

tp = 0
fp = 0
fn = 0
total = 0


for tag in tag_list:
    stream = np.array(drift_df[tag])
    labels = label_df[tag]
    detector = DetectorClass(**detector_params)

    if config['drift_detection']['reshape_streams']:
        stream = stream.reshape(stream.shape[0], 1)

    st = time.time()
    results = detector.detect_drift_window(stream)
    et = time.time()
    elapsed_time = et - st
    time_total += elapsed_time

    print('Results of Drift Detection:')
    print(f" Number of detected drifts {results['cnt_drift']}")
    for i in range(len(results['drift_ind'])):
        print(f" Drift detected at date: {drift_df['Timestamp'].iloc[results['drift_ind'][i]]}")
        print(f" With distance: {results['result_list'][i]}")

    start = -1
    end = -1
    results_ind = np.array(results['drift_ind'])


    for i in range(len(labels)):

        if (labels[i] == 0) & (i in results['drift_ind']):
            fp += 1

        if (labels[i] == 1) & (start == -1):
            start = i
        if (labels[i] == 1) & (start != -1):
            if labels[i + 1] == 0:
                end = i
                total += 1
                if len(np.where((results_ind >= start) & (results_ind <= end))[0]) != 0:
                    tp += 1
                else:
                    fn += 1
                start = -1
                end = -1

print(f"Average Execution time {time_total/len(config['drift_detection']['tag_list'])}")
print(f' Total number of drifts {total}')
print(f' True positives {tp}')
print(f' False positives {fp}')
print(f' False negatives {fn}')
