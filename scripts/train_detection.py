import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import argparse
import numpy as np
import pandas as pd
import os

from sklearn.utils.class_weight import compute_class_weight

from goodvibes.spectrum import *


MODEL = Sequential([
    Dense(128, activation='relu', input_shape=(8,)),  # Input layer with 8 input features
    Dense(64, activation='relu'),  # Hidden layer with 64 neurons
    Dense(32, activation='relu'),  # Another hidden layer
    Dense(9, activation='softmax')  # Output layer with 9 classes
])


def make_training_datasets(args):
    a, s = pipeline(args.data_dir)
    eigmodel = load_model(args.path_to_eig_model)
    Xa, ya = [], []
    for target in a.keys():
        for run in a[target].keys():
            for node in a[target][run].keys():
                for frame in a[target][run][node]['power']:
                    Xa.append(project(frame, eigmodel['acoustic_eigvectors'], eigmodel['acoustic_spectras_avg']))
                    ya.append(target)
    Xs, ys = [], []
    for target in s.keys():
        for run in s[target].keys():
            for node in s[target][run].keys():
                for frame in s[target][run][node]['power']:
                    Xs.append(project(frame, eigmodel['seismic_eigvectors'], eigmodel['seismic_spectras_avg']))
                    ys.append(target)
    Xa, ya = np.array(Xa), np.array(ya)
    Xs, ys = np.array(Xs), np.array(ys)
    return Xa, ya, Xs, ys


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, default='data')
    parser.add_argument('-o', '--output_dir', type=str, default='output')
    parser.add_argument('-e', '--path_to_eig_model', type=str, default='models/target_eig_model')
    args = parser.parse_args()
    run(args)