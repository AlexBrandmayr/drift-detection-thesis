#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Description:
#  File to run a series of (univariate) drift detection experiments on a dataframe and save the results
# -----------------------------------------------------------------------------------------------------------
from plots import *
import pandas as pd
import numpy as np
from src import *
import time
import json
import importlib
import sys

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

for tag in tag_list:
    stream = np.array(drift_df[tag])
    detector = DetectorClass(**detector_params)

    if config['drift_detection']['reshape_streams']:
        stream = stream.reshape(stream.shape[0], 1)

    st = time.time()
    results = detector.detect_drift_window(stream)
    et = time.time()
    elapsed_time = et - st
    time_total+=elapsed_time

    print('Results of Drift Detection:')
    print(f" Number of detected drifts {results['cnt_drift']}")
    for i in range(len(results['drift_ind'])):
        print(f" Drift detected at date: {drift_df['Timestamp'].iloc[results['drift_ind'][i]]}")
        print(f" With distance: {results['result_list'][i]}")


    if config['drift_detection']['create_plots']:
        save_path = config['drift_detection']['plot_path'] + config['drift_detection']['title']+'_'+tag+'_.png'
        plot_drift_plotly(drift_df, tag, results['drift_ind'], save_path, config['drift_detection']['line_plot'])

    if config['drift_detection']['create_reports']:
        report_name = config['drift_detection']['report_path'] + config['drift_detection']['title']+'_'+tag+'_.txt'
        create_report(drift_df, tag, results, elapsed_time, report_name, config)

print(f"Average Execution time {time_total/len(config['drift_detection']['tag_list'])}")
