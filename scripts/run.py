import argparse
import csv
from datetime import timedelta
import sys

from tensorflow.keras.models import load_model

from goodvibes.models import ClassifierModel, DetectionModel
from goodvibes.spectrum import load_model as load_spectrum_model
from goodvibes.utils.gv_datatypes import GvRecord, parseTimeFromMadsFilename
from goodvibes.utils.output import process_frame, read_data_in_chunks



if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('acoustic_file', help='acoustic file to process')
    parser.add_argument('seismic_file', help='seismic file to process')
    parser.add_argument('output_file', nargs='?', default=None, help='output file')
    parser.add_argument('--with-titles', help='include field titles in output (makes bigger files, but easier to read)', action='store_true', default=False)
    parser.add_argument('-e', '--eigmodel', type=str, help='Path to the eigmodel', default='models/target_eig_model')
    parser.add_argument('-d', '--detection_models', type=str, help='Path to the detection models', default='models/detect')
    parser.add_argument('-c', '--classifier_models', type=str, help='Path to the classifier models', default='models/classify')
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
    
    # Load models
    eigmodel = load_spectrum_model(args.eigmodel)
    detection_model = DetectionModel(args.detection_models)
    classifier_model = ClassifierModel(args.classifier_models)
    labels = classifier_model.labels

    # Start writing the output file.
    with open(output_filename, 'w', newline='') if output_filename is not None else sys.stdout as csv_file:
        csv_writer = csv.writer(csv_file)
        GvRecord.write_csv_header(csv_writer, with_titles)

        # Now we read the acoustic and seismic data from the input files a second at a time,
        # process the data, and then write the results to a file.
        offset_secs = 0
        for acoustic_data, seismic_data in read_data_in_chunks(acoustic_filename, seismic_filename, chunk_duration_ms=1000):
            # Calculate "wall clock time" for our current position in the file, based on the start time
            # (from the filename) plus the current offset.
            time_in_file = acoustic_startTime + timedelta(seconds=offset_secs)

            # Process data
            result = process_frame(
                acoustic_startTime,
                offset_secs,
                acoustic_data,
                seismic_data,
                eigmodel,
                detection_model,
                classifier_model,
                labels
                )

            # Write result to output file
            result.write_to_csv(csv_writer, offset_secs, time_in_file, with_titles)
            offset_secs = offset_secs + 1
