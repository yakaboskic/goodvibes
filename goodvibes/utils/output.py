#!/usr/bin/env python
import csv
import array
import argparse
import sys
import wave
from datetime import datetime, timedelta

from tqdm import tqdm

from goodvibes.utils.gv_datatypes import *
from goodvibes.spectrum import *

def read_data_in_chunks(filename_acoustic : str, filename_seismic: str, chunk_duration_ms=1000):
    # Open the WAV file
    with wave.open(filename_acoustic, 'rb') as wav_acoustic, wave.open(filename_seismic, 'rb') as wav_seismic:
        # Get the frame rate 
        frame_rate_acoustic = wav_acoustic.getframerate()
        frame_rate_seismic = wav_seismic.getframerate()

        # Calculate the number of frames per chunk
        frames_per_chunk_acoustic = int(frame_rate_acoustic * (chunk_duration_ms / 1000))
        frames_per_chunk_seismic = int(frame_rate_seismic * (chunk_duration_ms / 1000))

        # Continue reading until we reach the end of the file
        while True:
            # Read a chunk of frames
            frames_acoustic = wav_acoustic.readframes(frames_per_chunk_acoustic)
            frames_seismic = wav_seismic.readframes(frames_per_chunk_seismic)
            if len(frames_acoustic) == 0 or len(frames_seismic) == 0:
                break  # End of file

            # The wav files contain 16 bit samples, so convert the bytes to int16's
            data_acoustic = array.array('h')
            data_acoustic.frombytes(frames_acoustic)
            data_seismic = array.array('h')
            data_seismic.frombytes(frames_seismic)

            yield data_acoustic, data_seismic

def load_models(path_to_eig_model: str):
    # Load the models from disk
    eigmodel = load_model(path_to_eig_model)
    return eigmodel

def process_frame(
        start_time: datetime,
        offset_secs: int,
        acoustic_data: array,
        seismic_data: array,
        eigmodel,
        detection_model,
        classifer_model,
        labels,
        ) -> GvResult:
    a, s = pipeline_data(acoustic_data, seismic_data)
    # Produce eigen residuals for all classes
    Xa = []
    for spectrum in a:
        proj = project(spectrum, eigmodel['acoustic_eigvectors'], eigmodel['acoustic_spectras_avg'])
        Xa.append(np.array([proj[key] for key in labels]))
    Xs = []
    for spectrum in s:
        proj = project(spectrum, eigmodel['seismic_eigvectors'], eigmodel['seismic_spectras_avg'])
        Xs.append(np.array([proj[key] for key in labels]))
    Xa, Xs = np.array(Xa), np.array(Xs)
    # Predict detection
    detection = detection_model.predict(Xa, Xs)
    if detection:
        # Predict class
        classifer = classifer_model.predict(Xa, Xs)
        target_class = np.argmax(classifer)
    
    z = ZeroOrderSeismicFeatureData()

    d = DetectionData()
    d.detectionDeclaration = int(detection)

    tc = SecondOrderTargetCharacteristicsData()
    if detection:
        tc.targetIdentifier = target_class

    return GvResult(zeroOrderSeismic=z, detection=d, secondOrderTargetCharacteristics=tc)

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('acoustic_file', help='acoustic file to process')
    parser.add_argument('seismic_file', help='seismic file to process')
    parser.add_argument('output_file', nargs='?', default=None, help='output file')
    parser.add_argument('--with-titles', help='include field titles in output (makes bigger files, but easier to read)', action='store_true', default=False)
    args = parser.parse_args()

    acoustic_filename = args.acoustic_file
    seismic_filename = args.seismic_file
    output_filename = args.output_file
    with_titles = args.with_titles

    # Get file start times out of the file names
    acoustic_startTime = parseTimeFromMadsFilename(acoustic_filename)
    seismic_startTime = parseTimeFromMadsFilename(seismic_filename)

    if acoustic_startTime != seismic_startTime:
        raise("Files have different start times")

    # Start writing the output file.
    with open(output_filename, 'w', newline='') if output_filename is not None else sys.stdout as csv_file:
        csv_writer = csv.writer(csv_file)
        GvRecord.write_csv_header(csv_writer, with_titles)

        # Now we read the acoustic and seismic data from the input files a second at a time,
        # process the data, and then write the results to a file.
        offset_secs = 0
        for acoustic_data, seismic_data in tqdm(read_data_in_chunks(acoustic_filename, seismic_filename, chunk_duration_ms=1000)):
            # Calculate "wall clock time" for our current position in the file, based on the start time
            # (from the filename) plus the current offset.
            time_in_file = acoustic_startTime + timedelta(seconds=offset_secs)

            # Process data
            result = process_frame(acoustic_startTime, offset_secs, acoustic_data, seismic_data)

            # Write result to output file
            result.write_to_csv(csv_writer, offset_secs, time_in_file, with_titles)
            offset_secs = offset_secs + 1
