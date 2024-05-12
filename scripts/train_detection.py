import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio
import numpy as np
import pandas as pd
import os

from sklearn.utils.class_weight import compute_class_weight

# Constants
SR_A = 9866  # Sample rate to which all files will be converted
SR_S = 1550  

@tf.function
def load_wav_mono(wav_path, sr):
    # Read the wav file, this will load the full audio content into memory
    wav_data, sample_rate = tf.audio.decode_wav(
        contents=tf.io.read_file(wav_path),
        desired_channels=1)
    wav_data = tf.squeeze(wav_data, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav_data = tfio.audio.resample(wav_data, rate_in=sample_rate, rate_out=sr)
    return wav_data

def build_training_csv(args):
    clean_close = os.path.join(args.data_dir, 'clean-close')
    noise = os.path.join(args.data_dir, 'noise')
    df_clean_close_acoustic = pd.read_csv(os.path.join(clean_close, 'acoustic', 'clean-close.csv'))
    df_clean_close_seismic = pd.read_csv(os.path.join(clean_close, 'seismic', 'clean-close.csv'))
    df_noise_acoustic = pd.read_csv(os.path.join(noise, 'acoustic', 'noise.csv'))
    df_noise_seismic = pd.read_csv(os.path.join(noise, 'seismic', 'noise.csv'))
    df_clean_close_acoustic['label'] = 1
    df_clean_close_seismic['label'] = 1
    df_noise_acoustic['label'] = 0
    df_noise_seismic['label'] = 0
    df_clean_close_acoustic['filename'] = df_clean_close_acoustic['filename'].apply(lambda x: os.path.join(clean_close, 'acoustic', x))
    df_clean_close_seismic['filename'] = df_clean_close_seismic['filename'].apply(lambda x: os.path.join(clean_close, 'seismic', x))
    df_noise_acoustic['filename'] = df_noise_acoustic['filename'].apply(lambda x: os.path.join(noise, 'acoustic', x))
    df_noise_seismic['filename'] = df_noise_seismic['filename'].apply(lambda x: os.path.join(noise, 'seismic', x)) 
    df_clean_close_acoustic.drop(columns=list(set(list(df_clean_close_acoustic.columns)) - set(['filename', 'mode', 'label'])), inplace=True)
    df_clean_close_seismic.drop(columns=list(set(list(df_clean_close_seismic.columns)) - set(['filename', 'mode', 'label'])), inplace=True)
    df_noise_acoustic.drop(columns=list(set(list(df_noise_acoustic.columns)) - set(['filename', 'mode', 'label'])), inplace=True)
    df_noise_seismic.drop(columns=list(set(list(df_noise_seismic.columns)) - set(['filename', 'mode', 'label'])), inplace=True)
    df = pd.concat([df_clean_close_acoustic, df_clean_close_seismic, df_noise_acoustic, df_noise_seismic])
    class_weights = compute_class_weight('balanced', classes=np.unique(df['label']), y=df['label'])
    class_weights = dict(enumerate(class_weights))
    return df, class_weights

def load_acoustic_wav_for_map(filename, label):
    return load_wav_mono(filename, SR_A), label

def load_seismic_wav_for_map(filename, label):
    return load_wav_mono(filename, SR_S), label


def run(args):
    # Load training csv
    df, class_weights = build_training_csv(args)
    print(class_weights)

    # Acoustic Dataset
    df_acoustic = df[df['mode'] == 'acoustic']
    dataset_acoustic = tf.data.Dataset.from_tensor_slices((df_acoustic['filename'], df_acoustic['label']))
    dataset_acoustic = dataset_acoustic.map(load_acoustic_wav_for_map)
    # Seismic Dataset
    df_seismic = df[df['mode'] == 'seismic']
    dataset_seismic = tf.data.Dataset.from_tensor_slices((df_seismic['filename'], df_seismic['label']))
    dataset_seismic = dataset_seismic.map(load_seismic_wav_for_map)

    # Load the model.
    model = hub.load('https://www.kaggle.com/models/google/yamnet/TensorFlow2/yamnet/1')
    
    def extract_embedding(wav_data, label):
        scores, embeddings, spectrogram = model(wav_data)
        num_embeddings = tf.shape(embeddings)[0]
        #label_onehot = tf.one_hot(label, 2)
        return (embeddings, tf.repeat(label, num_embeddings))
    
    # Extract embeddings
    dataset_acoustic_embeddings = dataset_acoustic.map(extract_embedding).unbatch().shuffle(df_acoustic.shape[0])
    dataset_seismic_embeddings = dataset_seismic.map(extract_embedding).unbatch().shuffle(df_seismic.shape[0])


    # Create fine tuning model for acoustic and seismic data
    ft_model_acoustic = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1024,), dtype=tf.float32, name='input_embedding'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1),
    ],
    name='ft_model_acoustic'
    )

    ft_model_acoustic.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryFocalCrossentropy(
            from_logits=False,
            ),
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    callbacks = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
    history = ft_model_acoustic.fit(
        dataset_acoustic_embeddings.batch(32),
        epochs=20,
        callbacks=[callbacks],
        #class_weight=class_weights
    )
    return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, default='data')
    args = parser.parse_args()
    run(args)