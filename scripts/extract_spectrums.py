import argparse
import pickle
from goodvibes.spectrum import *

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
    # Test
    test_3a = a['2D'][1]['power'][0]
    print(project(test_3a, aeigvectors, a_avg))
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract spectras')
    parser.add_argument('-d', '--path_to_target_dir', type=str, help='Path to the target directory', default='data/targets/close')
    parser.add_argument('-o', '--output_folder', type=str, help='Path to the output folder', default='data/targets')
    args = parser.parse_args()
    run(args)