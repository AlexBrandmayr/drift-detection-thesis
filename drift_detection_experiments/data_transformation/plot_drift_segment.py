#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Description:
# Python file to visualize the drift segment
# -----------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import plotly.graph_objects as go



def plot_timeseries_with_color(x_axis, timeseries, color_stream, name,save_plot=False, save_path = None):
    """
    Plots a timeseries with color of points determined by another stream of zeros and ones.

    Parameters:
    -- timeseries (list): List of values representing the timeseries data.
    -- color_stream (list): List of values (0 or 1) representing the color stream data.
    -- savepath (str): Path where to save the plot

    Returns:
    -- None
    """
    # Create a scatter plot
    fig = go.Figure()

    # Add scatter trace for timeseries data as a line graph
    fig.add_trace(go.Scatter(x=x_axis, y=timeseries, mode='lines', name='Timeseries', showlegend=False))

    # Get indices where color_stream values are 1
    color_indices = [i for i, color_value in enumerate(color_stream) if color_value == 1]

    # Add scatter trace for color_stream data as markers
    fig.add_trace(go.Scatter(x=x_axis[color_indices], y=[timeseries[i] for i in color_indices],
                             mode='markers', marker=dict(color='red'), showlegend=False))

    # Get indices where color_stream values are 0
    color_indices = [i for i, color_value in enumerate(color_stream) if color_value == 0]

    # Add scatter trace for color_stream data as markers
    fig.add_trace(go.Scatter(x=x_axis[color_indices], y=[timeseries[i] for i in color_indices],
                             mode='markers', marker=dict(color='blue'), showlegend=False))

    # Show the plot
    if save_plot:
        fig.write_image(name, width=1800, height=500)
    else:
        # Show the plot
        fig.show()



DATA_SET = "df_drift_EI8"
TAG = "motor_current8.1"
DRIFT_START_POINTS = ['2020-03-1','2020-07-25']
DRIFT_END_POINTS = ['2020-04-8','2020-08-20']
SAVE_PATH = None

df = pd.read_pickle(DATA_SET)
df = df[["Timestamp", TAG]]

df['Timestamp'] = pd.to_datetime(df['Timestamp'])
labels = np.zeros(shape=len(df[TAG]))

for i, sp in enumerate(DRIFT_START_POINTS):
    start_date = pd.Timestamp(sp, tz='UTC')
    end_date = pd.Timestamp(DRIFT_END_POINTS[i], tz='UTC')
    indices = df[(df['Timestamp'] >= start_date) & (df['Timestamp'] <= end_date)].index
    labels[indices] = 1


plot_timeseries_with_color(df['Timestamp'], df[TAG], labels, TAG+'_label.png')

