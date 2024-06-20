import json
import numpy as np
import argparse
import os

from tqdm import tqdm

from goodvibes.utils.data import *
from goodvibes.signal import *

def run(args):
    # Read in files
    sig_info = read_signature_information(args.signature_information)
    run_info = read_target_run_log(args.run_log)
    emplacement = read_emplacement_information(args.emplacement_information)

    for run_idx, run in tqdm(run_info.iterrows(), total=run_info.shape[0]):
        placements = emplacement[emplacement['location'] == run['location']]
        run_placements = placements[placements['date'] == run['date']]
        gps_log_file = run['gps-log-file']
        gps_data = read_gps_log(os.path.join(args.position_folder, gps_log_file), start_date=run['start-datetime'], end_date=run['stop-datetime'])
        for node in run_placements['node-id'].unique():
            node_placement = run_placements[run_placements['node-id'] == node]
            # Read GPS logs of run and node distance
            gps_data['distance'] = gps_data.apply(lambda row: geodesic((row['latitude'], row['longitude']), (node_placement['lat'].iloc[0], node_placement['lon'].iloc[0])).meters, axis=1)
            if args.delta is not None:
                time_of_closest_approach = gps_data[gps_data['distance'] == gps_data['distance'].min()]['datetime'].values[0]
                start_datetime = time_of_closest_approach - pd.Timedelta(seconds=args.delta)
                stop_datetime = time_of_closest_approach + pd.Timedelta(seconds=args.delta)
                if start_datetime < run['start-datetime']:
                    start_datetime = run['start-datetime']
                if stop_datetime > run['stop-datetime']:    
                    stop_datetime = run['stop-datetime']
            else:
                start_datetime = run['start-datetime']
                stop_datetime = run['stop-datetime']
            print(start_datetime, stop_datetime)
            # Read acoustic data of node for run
            try:
                adata, arate, _ = read_wav(
                    sig_info,
                    node,
                    run['location'],
                    start_datetime=start_datetime,
                    stop_datetime=stop_datetime,
                    signatures_root_dir=args.signatures_folder,
                    mode='acoustic',
                )
                # Read seismic data of node for run
                sdata, srate, _ = read_wav(
                    sig_info,
                    node,
                    run['location'],
                    start_datetime=start_datetime,
                    stop_datetime=stop_datetime,
                    signatures_root_dir=args.signatures_folder,
                    mode='seismic',
                )
            except ValueError as e:
                print(f'Problem on run {run_idx} node {node}')
                print(run)
                print(node_placement)
                print(e)
                continue
            # Save the combined audio
            base_filename = f'{run["target-id"]}.{run_idx}.{node}'
            metadata = {
                'run_idx': int(run_idx),
                'node_id': int(node),
                'location': run['location'],
                'start_datetime': start_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                'stop_datetime': stop_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                'acoustic_rate': int(arate),
                'seismic_rate': int(srate),
            }
            # Add timestamps to all samples
            atimestamps = []
            for i in range(adata.shape[0]):
                atimestamps.append(start_datetime.timestamp() + i / arate)
            stimestamps = []
            for i in range(sdata.shape[0]):
                stimestamps.append(start_datetime.timestamp() + i / srate)
            adata = np.array([np.array(atimestamps), adata]).T
            sdata = np.array([np.array(stimestamps), sdata]).T
            np.save(os.path.join(args.output_folder, f'{base_filename}.acoustic.npy'), adata)
            np.save(os.path.join(args.output_folder, f'{base_filename}.seismic.npy'), sdata)
            distance = np.array([
                gps_data['datetime'].apply(lambda x: x.timestamp()).values,
                gps_data['distance'].values,
            ]).T
            np.save(os.path.join(args.output_folder, f'{base_filename}.distance.npy'), distance)
            with open(os.path.join(args.output_folder, f'{base_filename}.metadata.json'), 'w') as f:
                json.dump(metadata, f)
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract in range signals')
    parser.add_argument('-o', '--output_folder', type=str, help='Path to the output folder', default='data/targets')
    parser.add_argument('-s', '--signature_information', type=str, help='Path to the signature information CSV file', default='data/signature_information.csv')
    parser.add_argument('-r', '--run_log', type=str, help='Path to the target run log CSV file', default='data/target-run-log.csv')
    parser.add_argument('-e', '--emplacement_information', type=str, help='Path to the emplacement information CSV file', default='data/emplacement_information.csv')
    parser.add_argument('-p', '--position_folder', type=str, help='Path to the position folder', default='data/position/')
    parser.add_argument('-a', '--signatures_folder', type=str, help='Path to the signatures folder', default='data/signatures/')
    parser.add_argument('-d', '--delta', type=float, help='Symettric time delta around point of closest approach in seconds.', default=None)
    args = parser.parse_args()
    run(args)