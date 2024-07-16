import argparse
import csv
from datetime import timedelta
import sys

from tensorflow.keras.models import load_model

from goodvibes.models import ClassifierModel, DetectionModel
from goodvibes.spectrum import load_model as load_spectrum_model
from goodvibes.utils.gv_datatypes import GvRecord, parseTimeFromMadsFilename
from goodvibes.utils.output import process_frame_test, read_data_in_chunks



if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('acoustic_file', help='acoustic file to process')
    parser.add_argument('seismic_file', help='seismic file to process')
    parser.add_argument('output_file', nargs='?', default=None, help='output file')
    parser.add_argument('-o', '--offset_secs', type=int, help='Number of seconds to offset.', default=1)
    parser.add_argument('-e', '--eigmodel', type=str, help='Path to the eigmodel', default='models/target_eig_model')
    parser.add_argument('-d', '--detection_models', type=str, help='Path to the detection models', default='models/detect')
    parser.add_argument('-c', '--classifier_models', type=str, help='Path to the classifier models', default='models/classify')
    parser.add_argument('-s', '--signature_information', type=str, help='Path to the signature information CSV file', default='data/signature_information.csv')
    parser.add_argument('-r', '--run_log', type=str, help='Path to the target run log CSV file', default='data/target-run-log.csv')
    parser.add_argument('-n', '--emplacement_information', type=str, help='Path to the emplacement information CSV file', default='data/emplacement_information.csv')
    parser.add_argument('-p', '--position_folder', type=str, help='Path to the position folder', default='data/position/')
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
        csv_writer.writerow(['detection', 'target_id', 'true_distance'])

        # Now we read the acoustic and seismic data from the input files a second at a time,
        # process the data, and then write the results to a file.
        offset_secs = 0
        chunk_duration_ms = 1000 * args.offset_secs
        for acoustic_data, seismic_data in read_data_in_chunks(acoustic_filename, seismic_filename, chunk_duration_ms=chunk_duration_ms):
            # Calculate "wall clock time" for our current position in the file, based on the start time
            # (from the filename) plus the current offset.
            start_datetime = acoustic_startTime + timedelta(seconds=offset_secs)
            end_date = start_datetime + timedelta(seconds=offset_secs)
            # Process data
            detection, target_class, distance = process_frame_test(
                start_datetime,
                end_datetime,
                acoustic_data,
                seismic_data,
                eigmodel,
                detection_model,
                classifier_model,
                labels,
                acoustic_filename,
                run_info,
                emplacement_info,
                sig_info,
                gps_log_path=args.position_folder,
                )

            # Write result to output file
            csv_writer.writerow([start_datetime, end_datetime, detection, target_class, distance])
            offset_secs = offset_secs + args.offset_secs
