# Master Thesis Repository: Concept Drift Detection in Fiber Production Data

Welcome to the repository for my Master Thesis, focusing on the challenges and solutions associated with handling concept drifts in fiber production data. This repository contains code examples and implementations used in my research.

## Overview

This repository provides:

- Exploratory data analysis examples applied to measurement data from the fiber manufacturing industry.
- Code for an experimental framework to perform drift detection experiments on process data.

## Structure

- **drift_detection_experiments**: Contains all code necessary for drift detection experiments.
  - **concept_drift_detection**: Houses implementations of drift detection methods used in the thesis. These implementations are structured as Python classes, leveraging libraries like `river` and `frouros` designed explicitly for drift detection. Additionally, standard Python libraries such as `numpy` and `scipy.stats` are used where necessary.

- **run_experiment**: Contains files to execute drift detection experiments.
  - **config.json**: Configure experiment parameters, including data paths, result storage locations, and selected drift detection methods.
  - **run_experiment**: Run drift detection experiments, visualize results, and save them. Can also be used to test a method on multiple data series using `series_config.json` and `run_experiment_series`.
  - **evaluate_detector**: Evaluate a detector's performance on labeled data, calculating correctly detected drifts, false alarms, and missed drifts.

- **auxiliary_files**: Contains additional code files:
  - **plots.py** and **src.py**: Contain auxiliary functions for drift detection experiments.
  - **data_transformation**: Contains code files used for manual data labeling.

## Dependencies

- **requirements.txt** and **environment.yaml** contain all package dependencies needed to execute the code published here. You can use these files to set up your environment.
