import os
import re
import datetime
import xml.etree.ElementTree as ET
import pandas as pd


def process_sensor_xml(sensor_xml_path):
    """
    Process a sensor XML file into a more structured format.
    :param sensor_xml_path: str, path to the sensor XML file
    :return: list
    """
    tree = ET.parse(sensor_xml_path)
    root = tree.getroot()
    return {key: value for key, value in root.items()}

def organize_signatures(signature_path):
    """
    Organize a Signatures dataset into a more structured format.
    :param signature_path: str, path to the root of a signatures dataset
    :return: None
    """
    signature_header = [
        'location',
        'node-id', 
        'date', 
        'time', 
        'sampling_rate', 
        'mode', 
        'sequence-number', 
        'producer',
        'data-file',
        'up-time-seconds',
        'hw-config',
        'sw-mode',
        'status',
        'sensor-time-seconds',
        'sensor-time',
        'sample-rate-nominal',
        'sample-rate-actual',
        'lat',
        'lon',
        'alt-above-geoid-meters',
        'geoidal-separation-meters',
        'gps-fix-time-seconds',
        'gps-fix-time',
        'num-satellites',  
        ]
    # Get location from signature path
    location = signature_path.split('/')[-1].split('-')[0]
    # Initialize data list
    data = []

    # Walk directory and get all files
    for node_dir in os.listdir(signature_path):
        if os.path.isdir(os.path.join(signature_path, node_dir)):
            for date_dir in os.listdir(os.path.join(signature_path, node_dir)):
                if os.path.isdir(os.path.join(signature_path, node_dir, date_dir)):
                    for file in os.listdir(os.path.join(signature_path, node_dir, date_dir)):
                        if file.endswith('.xml'):
                            # Process XML file
                            xml_path = os.path.join(signature_path, node_dir, date_dir, file)
                            _data = process_sensor_xml(xml_path)
                            # Extract data from file name
                            split = re.split(r'[._-]', file)
                            node_id = int(split[0][4:])
                            mode = 'acoustic' if split[1] == 'Ch1' else 'seismic'
                            sequence_number = int(split[2])
                            dt = datetime.datetime(
                                year=int(split[3]),
                                month=int(split[4]),
                                day=int(split[5]),
                                hour=int(split[6]),
                                minute=int(split[7]),
                                second=int(split[8])
                            )
                            sample_rate = int(split[9][:-2])
                            # Create row
                            row = [
                                location,
                                node_id,
                                dt.date(),
                                dt.time(),
                                sample_rate,
                                mode,
                                sequence_number,
                                _data.get('producer', 'None'),
                                _data.get('dataFile', 'None'),
                                _data.get('uptimeSec', 'None'),
                                _data.get('hwConfig', 'None'),
                                _data.get('swMode', 'None'),
                                _data.get('status', 'None'),
                                _data.get('sensorTimeSec', 'None'),
                                _data.get('sensorTime', 'None'),
                                _data.get('sampleRateNominal', 'None'),
                                _data.get('sampleRateActual', 'None'),
                                _data.get('lat', 'None'),
                                _data.get('lon', 'None'),
                                _data.get('altAboveGeoidMeters', 'None'),
                                _data.get('geoidalSeparationMeters', 'None'),
                                _data.get('gpsFixTimeSec', 'None'),
                                _data.get('gpsFixTime', 'None'),
                                _data.get('numSats', 'None'),
                            ]
                            data.append(row)
    return pd.DataFrame.from_records(data, columns=signature_header)