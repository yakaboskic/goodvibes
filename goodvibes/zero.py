import librosa
import numpy as np


def extract_zero_order_features(
        data,
        rate,
        n_mfcc=13,
        n_fft=256,
        hop_length=512,
        n_mels=128,
        fmin=200.0,
        n_bands=6,
        ):
    """
    Extract zero order features from the data
    :param data: The audio data
    :param rate: The sample rate
    :param n_mfcc: The number of mfcc coefficients
    :param n_fft: The number of fft coefficients
    :param hop_length: The hop length
    :return: The zero order features
    """
    #print('mfccs')
    mfccs = librosa.feature.mfcc(y=data, sr=rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    #print(mfccs.shape)
    #print('chroma')
    chroma = librosa.feature.chroma_stft(y=data, sr=rate, n_fft=n_fft, hop_length=hop_length)
    #print(chroma.shape)
    #print('mel')
    mel = librosa.feature.melspectrogram(y=data, sr=rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    #print(mel.shape)
    #print('contrast')
    #contrast = librosa.feature.spectral_contrast(y=data, sr=rate, n_fft=n_fft, fmin=fmin, n_bands=n_bands)
    #print(contrast.shape)
    #print('zero_crossing_rate')
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=data)
    #print(zero_crossing_rate)
    #print('spectral centriod')
    spectral_centroid = librosa.feature.spectral_centroid(y=data, sr=rate, n_fft=n_fft, hop_length=hop_length)
    #print(spectral_centroid)
    #print('spectral_bandwidth')
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=data, sr=rate, n_fft=n_fft, hop_length=hop_length)
    #print(spectral_bandwidth)
    #print('spectral_rolloff')
    spectral_rolloff = librosa.feature.spectral_rolloff(y=data, sr=rate, n_fft=n_fft)
    #print(spectral_rolloff) 
    return {
        'mfccs': mfccs,
        'chroma': chroma,
        'mel': mel,
        #'contrast': contrast,
        'zero_crossing_rate': zero_crossing_rate,
        'spectral_centroid': spectral_centroid,
        'spectral_bandwidth': spectral_bandwidth,
        'spectral_rolloff': spectral_rolloff,
    }

def flatten_zero_order_features(features):
    """
    Flatten the zero order features into a feature array.
    :param features: The zero order features as output by extract_zero_order_features
    :return: The flattened feature array
    """
    # Flatten features and form a feature array
    labels = []
    data = []
    for key, arr in features.items():
        flatten = arr.flatten()
        labels.extend([f'{key}-{i}' for i in range(flatten.shape[0])])
        data.append(flatten)
    flatten = np.hstack(data)
    return flatten, labels

def stack_zero_order_features(features):
    """
    Stack the zero order features into a feature array.
    :param features: The zero order features as output by extract_zero_order_features
    :return: The flattened feature array
    """
    # Flatten features and form a feature array
    stack = np.vstack(
        (
            features['mfccs'],
            features['chroma'],
            features['mel'],
            #features['contrast'],
            features['zero_crossing_rate'],
            features['spectral_centroid'], 
            features['spectral_bandwidth'],
            features['spectral_rolloff'],
        )
    ).T
    return stack
