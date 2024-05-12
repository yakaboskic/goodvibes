import os
import numpy as np

from scipy.io import wavfile
from scipy.signal import resample

def read_wav(
        signature_info,
        node_id,
        location,
        run_id=None,
        target_run_log=None,
        mode='acoustic',
        start_datetime=None,
        stop_datetime=None,
        signatures_root_dir='data/signatures',
        sample_rate=None,
        ):
    """
    Read a wav file from the signatures directory
    :param signature_info: pd.DataFrame with the signature information
    :param node_id: int, the node id
    :param location: str, the location
    :param run_id: int, the run id
    :param target_run_log: pd.DataFrame, the run log
    :param mode: str, the mode
    :param start_datetime: datetime, the start datetime
    :param stop_datetime: datetime, the stop datetime
    :param signatures_root_dir: str, the root directory of the signatures
    :param sample_rate: int, the sample rate
    :return: np.array, the wav file
    :return: int, the sample rate
    """
    # Get the signature information for the node and location and mode
    sig_info = signature_info[
        (signature_info['node-id'] == node_id) &
        (signature_info['location'] == location) &
        (signature_info['mode'] == mode)
    ].sort_values('datetime')
    if start_datetime is not None and stop_datetime is not None:
        sig_info = sig_info[
            (sig_info['date'] == start_datetime.date()) &
            (sig_info['date'] == stop_datetime.date())
        ]
    if sig_info.shape[0] == 0:
        raise ValueError('No signature information found.')
    # Check to see if we are focused on a specific run
    if target_run_log is not None:
        run = target_run_log.iloc[run_id]
    else:
        run = None
    # Load the wav files
    _audio_data = []
    rates = []
    for _, row in sig_info.iterrows():
        wav = os.path.join(signatures_root_dir, row['data-file'])
        _rate, _data = wavfile.read(wav)
        _audio_data.append(_data)
        rates.append(_rate)
    if sample_rate is None:
        rate = max(rates)
    else:
        rate = sample_rate
    # Resample the data
    audio_data = []
    for data, _rate in zip(_audio_data, rates):
        if _rate != rate:
            data = resample(data, int(len(data) * rate / _rate))
        audio_data.append(data)
    audio_data = np.concatenate(audio_data)
    # Create a time axis in seconds
    time = np.linspace(0, len(audio_data) / rate, num=len(audio_data))
    # Get the audio start time for reference
    audio_start_datetime = sig_info['datetime'].iloc[0]
    # Set the start and stop times
    # Set defaults
    if start_datetime is None:
        start_index = 0
    if stop_datetime is None:
        stop_index = len(audio_data)
    if run is not None:
        _start_datetime = run['start-datetime'] - audio_start_datetime
        _stop_datetime = run['stop-datetime'] - audio_start_datetime
        start_seconds = start_datetime.total_seconds()
        stop_seconds = stop_datetime.total_seconds()
        start_index = int(start_seconds * rate)
        stop_index = int(stop_seconds * rate)
        if start_index < 0:
            start_index = 0
        if stop_index > len(audio_data):
            stop_index = len(audio_data)
    if start_datetime is not None:
        start_seconds = start_datetime - audio_start_datetime
        start_seconds = start_seconds.total_seconds()
        if start_seconds < 0:
            start_seconds = 0
        start_index = int(start_seconds * rate)
    else:
        start_index = 0
    if stop_datetime is not None:
        stop_seconds = stop_datetime - audio_start_datetime
        stop_seconds = stop_seconds.total_seconds()
        stop_index = int(stop_seconds * rate)
        if stop_index > len(audio_data):
            stop_index = len(audio_data)
    else:
        stop_index = len(audio_data)
    # Return the audio data
    return audio_data[start_index:stop_index], rate, time[start_index:stop_index]