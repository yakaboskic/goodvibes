import argparse
import pandas as pd
import os
import librosa
import numpy as np
import csv 

from tqdm import tqdm

from goodvibes.zero import extract_zero_order_features, flatten_zero_order_features
from goodvibes.utils.data import *

# Constants
SR_A = 9866  # Sample rate to which all files will be converted
FRAME_SIZE_A = 2048  # Frame size for the FFT
HOP_LENGTH_A = 512  # Number of samples between successive frames
N_MELS_A = 64  # Number of mel bands to generate
SR_S = 1550  
FRAME_SIZE_S = 256 
HOP_LENGTH_S = 512  
N_MELS_S = 64

def run(args):
    # Read in files
    sig_info = read_signature_information(args.signature_information)
    run_info = read_target_run_log(args.run_log)
    emplacement = read_emplacement_information(args.emplacement_information)
    
    # Extract zero order features from the clean-close data
    acoustic_meta = [
        pd.read_csv(os.path.join(args.noise, 'acoustic', 'noise.csv')),
        pd.read_csv(os.path.join(args.clean_close, 'acoustic', 'clean-close.csv')),
    ]
    seismic_meta = [
        pd.read_csv(os.path.join(args.noise, 'seismic', 'noise.csv')),
        pd.read_csv(os.path.join(args.clean_close, 'seismic', 'clean-close.csv')),
    ]

    is_header_written = False
    for j, (ameta, smeta) in enumerate(zip(acoustic_meta, seismic_meta)):
        for (_, row_a) ,(_, row_s) in tqdm(zip(ameta.iterrows(), smeta.iterrows()), total=ameta.shape[0]):
            if j == 0:
                filename_a = os.path.join(args.noise, 'acoustic', row_a['filename'])
                filename_s = os.path.join(args.noise, 'seismic', row_s['filename'])
            elif j == 1:
                filename_a = os.path.join(args.clean_close, 'acoustic', row_a['filename'])
                filename_s = os.path.join(args.clean_close, 'seismic', row_s['filename'])
            data_a, rate_a = librosa.load(filename_a, sr=SR_A)
            data_s, rate_s = librosa.load(filename_s, sr=SR_S)
            total_time = int(np.floor(data_a.shape[0] / rate_a))
            window_size_a = args.window_size * rate_a
            window_size_s = args.window_size * rate_s
            for i in range(0, total_time, args.window_size):
                if j == 0 and i >= args.max_iterations:
                    break
                i_a = i * rate_a
                i_s = i * rate_s
                #print('Working on Acoustic')
                #print(data_a[i_a:i_a+window_size_a].shape)
                features_a, labels_a = flatten_zero_order_features(
                    extract_zero_order_features(
                        data_a[i_a:i_a+window_size_a],
                        rate_a,
                        n_fft=FRAME_SIZE_A,
                        hop_length=HOP_LENGTH_A,
                        n_mels=N_MELS_A
                        )
                    )
                #print('Working on Seismic')
                #print(data_s[i_s:i_a+window_size_s].shape)
                features_s, label_s = flatten_zero_order_features(
                    extract_zero_order_features(
                        data_s[i_s:i_s+window_size_s],
                        rate_s,
                        n_fft=FRAME_SIZE_S,
                        hop_length=HOP_LENGTH_S,
                        n_mels=N_MELS_S
                        )
                    )
                features = list(features_a) + list(features_s)
                labels_a = [f'acoustic-{label}' for label in labels_a]
                label_s = [f'seismic-{label}' for label in label_s]
                header = labels_a + label_s
                header.insert(0, 'direction')
                header.insert(0, 'speed-throttle')
                header.insert(0, 'speed-kph')
                header.insert(0, 'make')
                header.insert(0, 'model')
                header.insert(0, 'year')
                header.insert(0, 'type')
                header.insert(0, 'target-id')
                header.insert(0, 'target-name')
                header.insert(0, 'location')
                header.insert(0, 'class')
                if j == 0:
                    features.insert(0, np.nan)
                    features.insert(0, np.nan)
                    features.insert(0, np.nan)
                    features.insert(0, np.nan)
                    features.insert(0, np.nan)
                    features.insert(0, np.nan)
                    features.insert(0, np.nan)
                    features.insert(0, np.nan)
                    features.insert(0, np.nan)
                    features.insert(0, row_a['location'])
                elif j == 1:
                    run = run_info.iloc[row_a['run-idx']] 
                    features.insert(0, run['direction'])
                    features.insert(0, run['speed-throttle'])
                    features.insert(0, run['speed-kph'])
                    features.insert(0, run['make'])
                    features.insert(0, run['model'])
                    features.insert(0, run['year'])
                    features.insert(0, run['type'])
                    features.insert(0, run['target-id'])
                    features.insert(0, run['target-name'])
                    features.insert(0, run['location'])
                features.insert(0, i)
                if is_header_written == False:
                    is_header_written = True
                    with open(args.output_file, 'w') as f:
                        writer = csv.writer(f)
                        writer.writerow(header)
                with open(args.output_file, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(features)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Construct Training Dataset')
    parser.add_argument('-o', '--output_file', type=str, help='Path to the output folder', default='output')
    parser.add_argument('-s', '--signature_information', type=str, help='Path to the signature information CSV file', default='data/signature_information.csv')
    parser.add_argument('-r', '--run_log', type=str, help='Path to the target run log CSV file', default='data/target-run-log.csv')
    parser.add_argument('-e', '--emplacement_information', type=str, help='Path to the emplacement information CSV file', default='data/emplacement_information.csv')
    parser.add_argument('-p', '--position_folder', type=str, help='Path to the position folder', default='data/position/')
    parser.add_argument('-a', '--signatures_folder', type=str, help='Path to the signatures folder', default='data/signatures/')
    parser.add_argument('-c', '--clean_close', type=str, help='Path to the clean vehicle data folder. Folder to use as training data for vehicle presense', default='data/clean-close')
    parser.add_argument('-n', '--noise', type=str, help='Path to the noise data folder. Folder to use as training data for no vehicle presense', default='data/noise')
    parser.add_argument('-w', '--window_size', type=int, help='The window size in seconds to use for the training data', default=1)
    parser.add_argument('-m', '--max-iterations', type=int, help='The maximum number of iterations to run', default=10)
    args = parser.parse_args()
    run(args)