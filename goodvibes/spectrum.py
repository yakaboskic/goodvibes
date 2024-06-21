import os, sys
import numpy as np
from collections import defaultdict


def normalize_amplitudes(data):
    """
    This function will normalize the amplitudes of the audio data to a zero mean.
    :param data: np.array
    :return: np.array
    """
    return data - np.mean(data)

def blockify(data, frame_size, hop_size):
    """
    This function will divide the data into blocks of size block_size.
    :param data: np.array
    :param frame_size: int
    :param hop_size: int
    :return: np.array
    """
    return [data[i:i+frame_size] for i in range(0, len(data), hop_size)]

def pipeline(path_to_target_dir):
    """
    This function is the main pipeline for the spectrum analysis.
    It will take the path to the target directory and return the
    results of the analysis.
    :param path_to_target_dir: str
    :return: dict
    """
    # Load files
    files = os.listdir(path_to_target_dir)
    acoustics = [f for f in files if 'acoustic' in f]
    seismic = [f for f in files if 'seismic' in f]

    # Process acoustic data
    acoustic_spectras = defaultdict(list)
    for f in acoustics:
        split = f.split('.')
        target_id, run_id, node_id, _, _ = split
        data = np.load(os.path.join(path_to_target_dir, f))
        if data.shape[0] == 0:
            continue
        if len(data.shape) > 1:
            data = data[:, 1]
        data = normalize_amplitudes(data)
        blocks = blockify(data, 2048, 1776)
        spectras = []
        for frame in blocks:
            # Apply the window to the frame
            windowed_frame = frame * np.hamming(len(frame))
            # Compute the FFT
            spectrum = np.fft.fft(windowed_frame)
            # Compute the magnitude of the spectrum
            magnitude = np.abs(spectrum)
            # Compute the power of the spectrum
            power = np.square(magnitude)[0:1024]
            if len(power) < 1024:
                continue
            # Normalize each frame to unit power
            power = power / np.sum(power)
            spectras.append(power)
        # Store the result
        acoustic_spectras[target_id].append({
            'run_id': run_id,
            'node_id': node_id,
            'power': spectras
        })

    # Process seismic data
    seismic_spectras = defaultdict(list)
    for f in seismic:
        split = f.split('.')
        target_id, run_id, node_id, _, _ = split
        data = np.load(os.path.join(path_to_target_dir, f))
        if data.shape[0] == 0:
            continue
        if len(data.shape) > 1:
            data = data[:, 1]
        data = normalize_amplitudes(data)
        blocks = blockify(data, 512, 448)
        spectras = []
        for frame in blocks:
            # Apply the window to the frame
            windowed_frame = frame * np.hamming(len(frame))
            # Compute the FFT
            spectrum = np.fft.fft(windowed_frame)
            # Compute the magnitude of the spectrum
            magnitude = np.abs(spectrum)
            # Compute the power of the spectrum
            power = np.square(magnitude)[0:512//2]
            if len(power) < 512//2:
                continue
            # Normalize each frame to unit power
            power = power / np.sum(power)
            spectras.append(power)
        # Store the result
        seismic_spectras[target_id].append({
            'run_id': run_id,
            'node_id': node_id,
            'power': spectras
        })
    return acoustic_spectras, seismic_spectras

def average_spectra(spectras):
    """
    This function will average the spectras for each target.
    :param spectras: dict
    :return: dict
    """
    # Extract all spectra
    result = defaultdict(list)
    for target_id, data in spectras.items():
        for res in data:
            result[target_id].extend(res['power'])
    # Average the spectra
    for target_id, data in result.items():
        result[target_id] = np.mean(data, axis=0)
    return result

def calculate_eigenvectors(average_spectra, spectras, num_eigenvectors=10):
    """
    This function will calculate the eigenvectors of the average spectra.
    :param average_spectra: dict
    :param spectras: dict
    :param num_eigenvectors: int
    :return: dict
    """
    result = {}
    for target_id, avg_spectral_vector in average_spectra.items():
        dist = []
        for res in spectras[target_id]:
            power = res['power']
            # Compute the distance
            dist.append(power - avg_spectral_vector)
        # Stack the data
        data = np.vstack(dist)
        # Compute the covariance matrix
        cov = np.cov(data)
        # Compute the eigenvectors
        eigvals, eigvecs = np.linalg.eig(cov)
        # Get the indices of the eigenvalues sorted in descending order
        sorted_indices = np.argsort(eigvals)[::-1]

        # Select the indices of the top M eigenvalues
        top_indices = sorted_indices[:num_eigenvectors]

        # Get the corresponding top M eigenvectors
        top_eigenvectors = eigvecs[:, top_indices]
        result[target_id] = top_eigenvectors
    return result

def project(spectrum, eigenvectors, avg_spectras):
    """
    This function will project the difference spectra onto the eigenvectors.
    :param spectrum: dict
    :param eigenvectors: dict
    :param avg_spectras: dict
    :return: dict
    """
    result = {}
    for target_id, avg_spectra in avg_spectras.items():
        # Norm adjust the spectrum
        norm_spectrum = spectrum - avg_spectra
        # Project onto eigenvector subspace
        norm_spectrum = np.expand_dims(norm_spectrum, axis=0)
        remainder = np.zeros(norm_spectrum.shape[1])
        for k in range(eigenvectors[target_id].shape[1]):
            eigvec_k = eigenvectors[target_id][:, k]
            eigvec_k = np.expand_dims(eigvec_k, axis=0)
            proj_k = np.dot(norm_spectrum.T, eigvec_k)
            wproj_k = np.squeeze(np.dot(proj_k, eigvec_k.T), axis=-1)
            remainder += wproj_k
        remainder = np.expand_dims(remainder, axis=0)
        # Calculate residual
        residual = norm_spectrum - remainder
        # Calculate magnitude of residual
        result[target_id] = np.linalg.norm(residual)
    return result

def make_prediction_vectors(spectras, eigvectors, avg_spectras):
    """
    This function will make the prediction vectors.
    :param spectras: spectrums we want to predict
    :param eigvectors: pretrained eigenvectors
    :param avg_spectras: pretrained average spectras
    :return: prediction vector, np.array
    :return: labels, np.array
    """
    predictions = []
    for spectrum in spectras:
        predictions.append(project(spectrum, eigvectors, avg_spectras))
    keys = sorted(predictions[0].keys())
    n = len(predictions)
    m = len(keys)
    result = np.zeros((n, m))
    for i, dct in enumerate(predictions):
            result[i] = [dct[key] for key in keys]
    return result, keys

def load_model(
        path_to_model: str
        ):
    """ Load the frequency eigenvector model.
    :param path_to_model: Path to the model dir.
    """
    with open(os.path.join(path_to_model, 'acoustic_eigvectors.pk'), 'rb') as f:
        acoustic_eigvectors = np.load(f)
    with open(os.path.join(path_to_model, 'seismic_eigvectors.pk'), 'rb') as f:
        seismic_eigvectors = np.load(f)
    with open(os.path.join(path_to_model, 'acoustic_spectras_avg.pk'), 'rb') as f:
        acoustic_spectras_avg = np.load(f)
    with open(os.path.join(path_to_model, 'seismic_spectras_avg.pk'), 'rb') as f:
        seismic_spectras_avg = np.load(f)
    return {
        'acoustic_eigvectors': acoustic_eigvectors,
        'seismic_eigvectors': seismic_eigvectors,
        'acoustic_spectras_avg': acoustic_spectras_avg,
        'seismic_spectras_avg': seismic_spectras_avg
    }