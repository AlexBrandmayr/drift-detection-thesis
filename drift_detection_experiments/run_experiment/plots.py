#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Description:
# Thi file contains functions to plot the results of drift detection experiments
# -----------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import plotly.express as px



def plot_drift(df, tag, stream, ind_list, save_path):
    plt.plot(df['Timestamp'], stream)
    plt.title(tag)
    for ind in ind_list:
        plt.axvline(x=df['Timestamp'].iloc[ind], color='red')
    plt.savefig(save_path)


def plot_drift_plotly(df, tag, drift_ind, save_path, line=False):
    if line:
        fig = px.line(df, x="Timestamp", y=tag)
    else:
        fig = px.scatter(df, x="Timestamp", y=tag)

    for ind in drift_ind:
        fig.add_vline(x=df['Timestamp'].iloc[ind])

    #fig.write_image(save_path, width=1800, height=500)
    fig.show()
