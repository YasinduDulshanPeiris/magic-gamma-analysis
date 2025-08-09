# MAGIC Gamma Telescope Data Analysis

This repository contains a Jupyter Notebook (`model.ipynb`) for analyzing the MAGIC Gamma Telescope dataset, which classifies gamma and hadron events based on telescope measurements. The notebook implements machine learning models, including k-Nearest Neighbors (kNN) and a neural network, to predict event classes.

## Dataset
The dataset is sourced from the UCI Machine Learning Repository:
- **Source**: Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
- **Donated by**: P. Savicky, Institute of Computer Science, AS of CR, Czech Republic.
- **Features**: 10 attributes (e.g., `fLength`, `fWidth`, `fSize`, etc.) describing telescope measurements.
- **Target**: Binary classification (`class`: 1 for gamma, 0 for hadron).

## Notebook Overview
- **Data Loading**: Loads `magic04.data` into a Pandas DataFrame.
- **Preprocessing**: Converts class labels to binary (gamma=1, hadron=0), scales features using `StandardScaler`, and oversamples the training set with `RandomOverSampler`.
- **Visualization**: Plots histograms of features for gamma and hadron classes.
- **Modeling**:
  - k-Nearest Neighbors (kNN) with `n_neighbors=5`.
  - Neural network with TensorFlow/Keras.
- **Evaluation**: Reports precision, recall, and F1-score for both models on the test set.
- **Results**:
  - kNN: ~82% accuracy (based on classification report).
  - Neural Network: ~87% accuracy (based on classification report).

## Requirements
To run the notebook, install the following Python packages:
```bash
pip install numpy pandas matplotlib scikit-learn imblearn tensorflow