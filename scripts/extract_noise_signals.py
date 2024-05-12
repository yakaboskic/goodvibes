import numpy as np
import argparse
import os
from scipy.io import wavfile

from tqdm import tqdm

from goodvibes.utils.data import *
from goodvibes.signal import *


def run(args):
    # Read in files
    sig_info = read_signature_information(args.signature_information)
    run_info = read_target_run_log(args.run_log)
    emplacement = read_emplacement_information(args.emplacement_information)

    # Get the gap times
    gaps = find_noise_times(sig_info, run_info)

    metadata = []
    for idx, row in tqdm(gaps.iterrows(), total=gaps.shape[0]):
        if row['stop'] == -1:
            stop = None
        else:
            stop = row['stop']
        try:
            data, rate, _ = read_wav(
                sig_info,
                row['node-id'],
                row['location'],
                start_datetime=row['start'],
                stop_datetime=stop,
                signatures_root_dir=args.signatures_folder,
                mode=args.mode,
            )
        except ValueError as e:
            print('Problem on gap', idx)
            print(row)
            print(e)
            break
        filename = f'node-{row["node-id"]}_location-{row["location"]}_noise-{idx}.wav'
        metadata.append([
            filename,
            row['node-id'],
            row['location'],
            row['date'],
            row['start'],
            row['stop'],
            rate,
            args.mode,
        ])
        # Save the combined audio
        wavfile.write(os.path.join(args.output_folder, filename), rate, data.astype(np.int16))
    # Save metadata
    pd.DataFrame(metadata, columns=['filename', 'node-id', 'location', 'date', 'start', 'stop', 'rate', 'mode']).to_csv(os.path.join(args.output_folder, 'noise.csv'), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract in range signals')
    parser.add_argument('-o', '--output_folder', type=str, help='Path to the output folder', default='output')
    parser.add_argument('-s', '--signature_information', type=str, help='Path to the signature information CSV file', default='data/signature_information.csv')
    parser.add_argument('-r', '--run_log', type=str, help='Path to the target run log CSV file', default='data/target-run-log.csv')
    parser.add_argument('-e', '--emplacement_information', type=str, help='Path to the emplacement information CSV file', default='data/emplacement_information.csv')
    parser.add_argument('-p', '--position_folder', type=str, help='Path to the position folder', default='data/position/')
    parser.add_argument('-a', '--signatures_folder', type=str, help='Path to the signatures folder', default='data/signatures/')
    parser.add_argument('-m', '--mode', type=str, help='Mode of .wav data, e.g., acoustic or seismic', default='acoustic')
    args = parser.parse_args()
    run(args)