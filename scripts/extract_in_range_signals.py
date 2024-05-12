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

    # Calculate in range times
    times = find_in_range_times(run_info, emplacement, delta=args.delta, gps_log_path=args.position_folder).dropna()

    metadata = []
    for i, row in tqdm(times.iterrows(), total=times.shape[0]):
        run = run_info.iloc[row['run-idx']]
        try:
            data, rate, _ = read_wav(
                sig_info,
                row['node-id'],
                run['location'],
                start_datetime=row['datetime-min'],
                stop_datetime=row['datetime-max'],
                signatures_root_dir=args.signatures_folder,
                mode=args.mode,
            )
        except ValueError:
            print('Problem on run', row['run-idx'])
            print(run)
            print(row)
            break
        filename = f'node-{row["node-id"]}_run-{row["run-idx"]}_delta-{args.delta}.wav'
        metadata.append([
            filename,
            row['node-id'],
            run['location'],
            row['run-idx'],
            row['datetime-min'],
            row['datetime-max'],
            args.delta,
            rate,
            args.mode,
        ])
        # Save the combined audio
        wavfile.write(os.path.join(args.output_folder, filename), rate, data.astype(np.int16))
    # Save metadata
    pd.DataFrame(metadata, columns=['filename', 'node-id', 'location', 'run-idx', 'datetime-min', 'datetime-max', 'delta', 'rate', 'mode']).to_csv(os.path.join(args.output_folder, 'clean-close.csv'), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract in range signals')
    parser.add_argument('-o', '--output_folder', type=str, help='Path to the output folder', default='output')
    parser.add_argument('-s', '--signature_information', type=str, help='Path to the signature information CSV file', default='data/signature_information.csv')
    parser.add_argument('-r', '--run_log', type=str, help='Path to the target run log CSV file', default='data/target-run-log.csv')
    parser.add_argument('-e', '--emplacement_information', type=str, help='Path to the emplacement information CSV file', default='data/emplacement_information.csv')
    parser.add_argument('-p', '--position_folder', type=str, help='Path to the position folder', default='data/position/')
    parser.add_argument('-a', '--signatures_folder', type=str, help='Path to the signatures folder', default='data/signatures/')
    parser.add_argument('-d', '--delta', type=int, help='Delta in meters', default=20)
    parser.add_argument('-m', '--mode', type=str, help='Mode of .wav data, e.g., acoustic or seismic', default='acoustic')
    args = parser.parse_args()
    run(args)
