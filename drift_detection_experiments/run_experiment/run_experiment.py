#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Description:
# File to run a drift detection experiment and save the results
# -----------------------------------------------------------------------------------------------------------
from plots import *
from src import *
import pandas as pd
import numpy as np
import time
import json
import importlib
import sys

with open('config.json') as f:
    config = json.load(f)

sys.path.append("C:/Users/brand/OneDrive/Dokumente/Studium/Master_Thesis_Sicherung/Code_project/drift_detection_experiments/concept_drift_detection")

tag = config['drift_detection']['tag']
df_name = config['drift_detection']['data_frame']

detector_module, detector_class = config['detector']['class'].rsplit('.', 1)
DetectorClass = getattr(importlib.import_module(detector_module), detector_class)

detector_params = config['detector']['params']
detector = DetectorClass(**detector_params)

drift_df = pd.read_pickle(df_name)
stream = np.array(drift_df.loc[:, tag])


if config['drift_detection']['reshape_stream']:
    stream = stream.reshape(stream.shape[0], 1)

st = time.time()
results = detector.detect_drift_window(stream)
et = time.time()
elapsed_time = et - st

print('Results of Drift Detection:')
print(f" Number of detected drifts {results['cnt_drift']}")
for i in range(len(results['drift_ind'])):
    print(f" Drift detected at date: {drift_df['Timestamp'].iloc[results['drift_ind'][i]]}")
    print(f" With distance: {results['result_list'][i]}")


if config['drift_detection']['create_plot']:
    save_path = config['drift_detection']['plot_path'] + config['drift_detection']['plot_name']
    plot_drift_plotly(drift_df, tag, results['drift_ind'], save_path, config['drift_detection']['line_plot'])

if config['drift_detection']['create_report']:
    report_name = config['drift_detection']['report_path'] + config['drift_detection']['report_name']
    create_report(drift_df, tag, results, elapsed_time, report_name, config)
