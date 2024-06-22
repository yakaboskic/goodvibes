import argparse
import pickle
import os, re

from scipy.io import wavfile
from goodvibes.spectrum import *

def build_noise_target_datasets(args):
    """
    This function will build the noise target datasets.
    :param noise_dir: str
    :return: dict, dict
    """
    for f in os.listdir(os.path.join(args.path_to_noise_dir, 'acoustic')):
    # Process acoustic data
        split = re.split(r'[_.-]+', f)
        if len(split) != 7:
            continue
        _, node_id, _, location_name, _, idx, _ = split
        rate,  data = wavfile.read(os.path.join(args.path_to_noise_dir, 'acoustic', f))
        data = data[:args.max_length_sec*rate]
        np.save(f'{args.output_folder}/{location_name}.{idx}.{node_id}.acoustic.npy', data)
    for f in os.listdir(os.path.join(args.path_to_noise_dir, 'seismic')):
        split = re.split(r'[_.-]+', f)
        if len(split) != 7:
            continue
        _, node_id, _, location_name, _, idx, _ = split
        rate, data = wavfile.read(os.path.join(args.path_to_noise_dir, 'seismic', f))
        data = data[:args.max_length_sec*rate]
        np.save(f'{args.output_folder}/{location_name}.{idx}.{node_id}.seismic.npy', data)

def run(args):
    """
    This function will run the pipeline with the given arguments.
    :param args: argparse.Namespace
    :return: None
    """
    # Run the pipeline
    a, s = pipeline(args.path_to_target_dir)
    # Save the results
    with open(f'{args.output_folder}/acoustic_spectras.pk', 'wb') as f:
        pickle.dump(a, f)
    with open(f'{args.output_folder}/seismic_spectras.pk', 'wb') as f:
        pickle.dump(s, f)
    # Average the results
    a_avg = average_spectra(a)
    s_avg = average_spectra(s)
    # Save the results
    with open(f'{args.output_folder}/acoustic_spectras_avg.pk', 'wb') as f:
        pickle.dump(a_avg, f)
    with open(f'{args.output_folder}/seismic_spectras_avg.pk', 'wb') as f:
        pickle.dump(s_avg, f)
    # Calculate the eigenvectors
    aeigvectors = calculate_eigenvectors(a_avg, a)
    seigvectors = calculate_eigenvectors(s_avg, s)
    with open(f'{args.output_folder}/acoustic_eigvectors.pk', 'wb') as f:
        pickle.dump(aeigvectors, f)
    with open(f'{args.output_folder}/seismic_eigvectors.pk', 'wb') as f:
        pickle.dump(seigvectors, f)
    print(aeigvectors)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract noise spectras')
    parser.add_argument('-d', '--path_to_noise_dir', type=str, help='Path to the noise directory', default='data/noise')
    parser.add_argument('-o', '--output_folder', type=str, help='Path to the output folder', default='data/targets')
    parser.add_argument('-t', '--path_to_target_dir', type=str, help='Path to the target directory', default='data/targets/noise')
    parser.add_argument('-m', '--max_length_sec', type=int, help='Max length in seconds', default=5)
    args = parser.parse_args()
    #build_noise_target_datasets(args)
    run(args)