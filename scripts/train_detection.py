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
        for item in a[target]:
            for spectras in item['power']:
                for frame in spectras:
                    Xa.append(project(frame, eigmodel['acoustic_eigvectors'], eigmodel['acoustic_spectras_avg']))
                    ya.append(target)
    Xs, ys = [], []
    for target in s.keys():
        for item in s[target]:
            for spectras in item['power']:
                for frame in spectras:
                    Xs.append(project(frame, eigmodel['seismic_eigvectors'], eigmodel['seismic_spectras_avg']))
                    ys.append(target)
    Xa, ya = np.array(Xa), np.array(ya)
    Xs, ys = np.array(Xs), np.array(ys)
    return Xa, ya, Xs, ys

def train_classifers(args):
    Xa, ya, Xs, ys = make_training_datasets(args)
    class_weights_a = compute_class_weight('balanced', classes=np.unique(ya), y=ya)
    class_weights_s = compute_class_weight('balanced', classes=np.unique(ys), y=ys)
    amodel = MODEL.copy()
    smodel = MODEL.copy()
    amodel.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    amodel.fit(Xa, ya, class_weight=class_weights_a, epochs=10)
    amodel.save(os.path.join(args.output_dir, 'aclassifier.h5'))
    smodel.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    smodel.fit(Xs, ys, class_weight=class_weights_s, epochs=10)
    smodel.save(os.path.join(args.output_dir, 'sclassifier.h5'))
    return

def run(args):
    train_classifers(args)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, default='data/targets/close')
    parser.add_argument('-o', '--output_dir', type=str, default='output')
    parser.add_argument('-e', '--path_to_eig_model', type=str, default='models/target_eig_model')
    args = parser.parse_args()
    run(args)