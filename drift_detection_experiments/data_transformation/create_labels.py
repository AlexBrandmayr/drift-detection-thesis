#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Description:
# This python file is for creating labels for the drift segments
# -----------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import pickle

DATA_SET = "df_drift_EI3"
TAG_LIST = ["motor_current8.1",
            "motor_current8.2",
             "motor_current8.3"]
DRIFT_SEGMENTS = {
    "motor_current8.1": {
        "start_points": ['2020-11-01'],
        "end_points": ['2020-12-01']
    },
    "motor_current8.2": {
        "start_points": ['2022-06-28'],
        "end_points": ['2022-07-28']
    },
    "motor_current8.3": {
        "start_points": ['2022-09-25'],
        "end_points": ['2022-10-25']
    }
}
SAVE_PATH = '/path/to/your/data/labels3_df'

df = pd.read_pickle(DATA_SET)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])


df_labels = pd.DataFrame()

for tag in TAG_LIST:
    labels = np.zeros(shape=len(df['Timestamp']))

    for i, sp in enumerate(DRIFT_SEGMENTS[tag]['start_points']):
        start_date = pd.Timestamp(sp, tz='UTC')
        end_date = pd.Timestamp(DRIFT_SEGMENTS[tag]['end_points'][i], tz='UTC')
        indices = df[(df['Timestamp'] >= start_date) & (df['Timestamp'] <= end_date)].index
        labels[indices] = 1
        df_labels[tag] = labels


print(df_labels.describe())
with open(SAVE_PATH, 'wb') as f:
    pickle.dump(df_labels, f)