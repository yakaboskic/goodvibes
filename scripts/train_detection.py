import copy
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from kerastuner.tuners import RandomSearch

import argparse
import numpy as np
import pandas as pd
import os

from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from goodvibes.spectrum import *


MODEL = Sequential([
    Dense(128, activation='relu', input_shape=(8,)),  # Input layer with 8 input features
    Dense(64, activation='relu'),  # Hidden layer with 64 neurons
    Dense(32, activation='relu'),  # Another hidden layer
    Dense(8, activation='softmax')  # Output layer with 9 classes
])

DETECT_MODEL = Sequential([
    Dense(128, activation='relu', input_shape=(8,)),  # Input layer with 8 input features
    Dense(64, activation='relu'),  # Hidden layer with 64 neurons
    Dense(32, activation='relu'),  # Another hidden layer
    Dense(2, activation='softmax')  # Output layer with 9 classes
])

def make_training_datasets(args):
    a, s = pipeline(args.data_dir)
    keys = list(sorted(a.keys()))
    eigmodel = load_model(args.path_to_eig_model)
    Xa, ya = [], []
    for target in tqdm(a.keys()):
        for item in a[target]:
            for spectra in item['power']:
                    proj = project(spectra, eigmodel['acoustic_eigvectors'], eigmodel['acoustic_spectras_avg'])
                    Xa.append(np.array([proj[key] for key in keys]))
                    ya.append(keys.index(target))
    Xs, ys = [], []
    for target in tqdm(s.keys()):
        for item in s[target]:
            for spectra in item['power']:
                    proj = project(spectra, eigmodel['seismic_eigvectors'], eigmodel['seismic_spectras_avg'])
                    Xs.append(np.array([proj[key] for key in keys]))
                    ys.append(keys.index(target))
    Xa, ya = np.array(Xa), np.array(ya)
    Xs, ys = np.array(Xs), np.array(ys)
    with open(os.path.join(args.output_dir, 'acoustic-targets.pk'), 'wb') as f:
        pickle.dump((Xa, ya, keys), f)
    with open(os.path.join(args.output_dir, 'seismic-targets.pk'), 'wb') as f:
        pickle.dump((Xs, ys, keys), f)
    return Xa, ya, Xs, ys, keys

def make_noise_datasets(args):
    a, s = pipeline(args.noise_dir)
    eigmodel = load_model(args.path_to_eig_model)
    Xa = []
    keys = list(sorted(eigmodel['acoustic_eigvectors'].keys()))
    for target in tqdm(a.keys()):
        for item in a[target]:
            for spectra in item['power']:
                    proj = project(spectra, eigmodel['acoustic_eigvectors'], eigmodel['acoustic_spectras_avg'])
                    Xa.append(np.array([proj[key] for key in keys]))
    Xs= []
    for target in tqdm(s.keys()):
        for item in s[target]:
            for spectra in item['power']:
                    proj = project(spectra, eigmodel['seismic_eigvectors'], eigmodel['seismic_spectras_avg'])
                    Xs.append(np.array([proj[key] for key in keys]))
    Xa = np.array(Xa)
    Xs = np.array(Xs)
    with open(os.path.join(args.output_dir, 'acoustic-noise.pk'), 'wb') as f:
        pickle.dump((Xa, keys), f)
    with open(os.path.join(args.output_dir, 'seismic-noise.pk'), 'wb') as f:
        pickle.dump((Xs, keys), f)
    return Xa, Xs, keys


def train_classifers(args):
    #Xa, ya, Xs, ys, class_labels = make_training_datasets(args)
    with open(os.path.join(args.output_dir, 'acoustic-targets.pk'), 'rb') as f:
        Xa, ya, labels = pickle.load(f)
    with open(os.path.join(args.output_dir, 'seismic-targets.pk'), 'rb') as f:
        Xs, ys, _ = pickle.load(f)
    amodel = copy.copy(MODEL)
    smodel = copy.copy(MODEL)

    amodel.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    amodel.fit(Xa, ya, epochs=100)
    amodel.save(os.path.join(args.output_dir, 'aclassifier.keras'))
    smodel.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    smodel.fit(Xs, ys, epochs=100)
    smodel.save(os.path.join(args.output_dir, 'sclassifier.keras'))
    return

def train_detector(args):
    #Xan, Xsn, keys = make_noise_datasets(args)
    with open(os.path.join(args.output_dir, 'acoustic-noise.pk'), 'rb') as f:
        Xan, labels = pickle.load(f)
    with open(os.path.join(args.output_dir, 'seismic-noise.pk'), 'rb') as f:
        Xsn, _ = pickle.load(f)
    with open(os.path.join(args.output_dir, 'acoustic-targets.pk'), 'rb') as f:
        Xat, _, labels = pickle.load(f)
    with open(os.path.join(args.output_dir, 'seismic-targets.pk'), 'rb') as f:
        Xst, _, _ = pickle.load(f)

    Xa = np.concatenate([Xan, Xat])
    Xs = np.concatenate([Xsn, Xst])
    ya = np.concatenate([np.zeros(len(Xan)), np.ones(len(Xat))]) 
    ys = np.concatenate([np.zeros(len(Xsn)), np.ones(len(Xst))])
    amodel = copy.copy(DETECT_MODEL)
    amodel.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    amodel.fit(Xa, ya, epochs=100, shuffle=True)
    amodel.save(os.path.join(args.output_dir, 'acoustic_detector.keras'))
    smodel = copy.copy(DETECT_MODEL)
    smodel.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    smodel.fit(Xs, ys, epochs=100, shuffle=True)
    smodel.save(os.path.join(args.output_dir, 'seismic_detector.keras'))
    return

def run(args):
    train_classifers(args)
    #train_detector(args)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, default='data/targets/close')
    parser.add_argument('-n', '--noise_dir', type=str, default='data/targets/noise')
    parser.add_argument('-o', '--output_dir', type=str, default='')
    parser.add_argument('-e', '--path_to_eig_model', type=str, default='models/target_eig_model')
    args = parser.parse_args()
    run(args)