#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Description:
# File contains functions for the drift detection experiments
# -----------------------------------------------------------------------------------------------------------


def create_report(df, tag, results, exec_time, file_name, config):
    f = open(file_name, "w")
    f.write(f"Tagnamme {tag}")
    f.write('\n')
    f.write('\n')
    f.write(f"Start date {df['Timestamp'].iloc[0]}")
    f.write('\n')
    f.write('\n')
    f.write(f"End date {df['Timestamp'].iloc[-1]}")
    f.write('\n')
    f.write('\n')
    f.write(f"Length of data stream {len(df[tag])}")
    f.write('\n')
    f.write('\n')
    f.write(f"Number of detected drifts {results['cnt_drift']}")
    f.write('\n')
    f.write('\n')
    f.write(f"Execution time of drift detection {exec_time}")
    f.write('\n')
    f.write('\n')
    f.write(f"Alogrithm used {config['detector']['class']}")
    f.write('\n')
    f.write('\n')
    f.write(f"Alogrithm Parameters {config['detector']['params']}")
    f.write('\n')
    f.write('\n')
    for i in range(len(results['drift_ind'])):
        f.write(f" Drift detected at date: {df['Timestamp'].iloc[results['drift_ind'][i]]}")
        f.write(f" With distance: {results['result_list'][i]}")
        f.write('\n')

    f.close()
