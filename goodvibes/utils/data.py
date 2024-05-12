import os
import re
import datetime
import xml.etree.ElementTree as ET
import pandas as pd

from geopy.distance import geodesic


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
                            # Ensure associated wav file exists by removing xml extension
                            if not os.path.exists(os.path.join(signature_path, node_dir, date_dir, file[:-4])):
                                print(f'No associated wav file for {file}')
                                continue
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

def parse_nmea_sentence(sentence, mode='GPGGA'):
    """
    Parse an NMEA sentence and return a dictionary of the data. Will only parse $GPGGA sentences.
    
    Explanation of the Sentences:
    $GPGGA - Global Positioning System Fix Data. This includes:
        Time
        Latitude and longitude
        Fix quality (e.g., GPS fix, DGPS fix)
        Number of satellites being tracked
        Horizontal dilution of precision
        Altitude
        Geoidal separation
    $GPRMC - Recommended Minimum Specific GPS/Transit Data. This includes:
        Time
        Status (Active or Void)
        Latitude and longitude
        Speed over ground
        Course over ground
        Date
        Mode

    :param sentence: str, NMEA sentence
    :param mode: str, NMEA mode (GPGGA, GPRMC)
    :return: dict
    """
    parts = sentence.split(',')
    try:
        if sentence.startswith('$GPGGA') and mode == 'GPGGA':
            time = datetime.datetime.strptime(parts[1], '%H%M%S.%f')
            latitude = float(parts[2])/100
            lat_dir = parts[3]
            longitude = float(parts[4])/100
            long_dir = parts[5]
            return {
                'time': time.time(),
                'latitude': latitude,
                'lat_dir': lat_dir,
                'longitude': longitude,
                'long_dir': long_dir
                }
        elif sentence.startswith('$GPRMC') and mode == 'GPRMC':
            time = datetime.datetime.strptime(parts[1], '%H%M%S.%f')
            latitude = float(parts[3])/100
            lat_dir = parts[4]
            longitude = float(parts[5])/100
            long_dir = parts[6]
            speed = float(parts[7])
            course = float(parts[8])
            date = datetime.datetime.strptime(parts[9], '%d%m%y')
            return {
                'time': time.time(),
                'latitude': latitude,
                'lat_dir': lat_dir,
                'longitude': longitude,
                'long_dir': long_dir,
                'speed': speed,
                'course': course,
                'date': date.date(),
                }
    except Exception as e:
        print(f'Error parsing NMEA sentence: {e}')
        pass
    return None

def convert_to_decimal(degrees_minutes, direction):
    """
    Convert latitude and longitude from degrees and minutes to decimal degrees.
    :param degrees_minutes: float, latitude or longitude in degrees and minutes
    :param direction: str, direction (N, S, E, W)
    :return: float, decimal degrees
    """
    degrees = int(degrees_minutes)
    minutes = (degrees_minutes - degrees) * 100
    decimal_degrees = degrees + minutes / 60
    if direction in ['S', 'W']:
        decimal_degrees *= -1
    return decimal_degrees

def parse_nmea_data(data):
    """
    Parse NMEA data and return a DataFrame of the parsed data.
    :param data: str, NMEA data
    :return: pd.DataFrame
    """
    parsed_data_gpgga = [parse_nmea_sentence(line, 'GPGGA') for line in data if line.startswith('$GPGGA')]
    parsed_data_gprmc = [parse_nmea_sentence(line, 'GPRMC') for line in data if line.startswith('$GPRMC')]
    # Remove any None values
    parsed_data_gpgga = [d for d in parsed_data_gpgga if d]
    parsed_data_gprmc = [d for d in parsed_data_gprmc if d]
    # Merge the two datasets
    if len(parsed_data_gpgga) > 0 and len(parsed_data_gprmc) > 0:
        time_gpgga = {gpgga['time']: gpgga for gpgga in parsed_data_gpgga}
        time_gprmc = {gprmc['time']: gprmc for gprmc in parsed_data_gprmc}
        parsed_data = []
        for time in time_gpgga:
            if time in time_gprmc:
                parsed_data.append({**time_gpgga[time], **time_gprmc[time]})
            else:
                parsed_data.append(time_gpgga[time])
        for time in time_gprmc:
            if time not in time_gpgga:
                parsed_data.append(time_gprmc[time])
    elif len(parsed_data_gpgga) > 0:
        parsed_data = parsed_data_gpgga
    elif len(parsed_data_gprmc) > 0:
        parsed_data = parsed_data_gprmc
    # Create DataFrame
    df = pd.DataFrame(parsed_data)
    # Convert latitude and longitude to decimal degrees
    df['latitude'] = df.apply(lambda row: convert_to_decimal(row['latitude'], row['lat_dir']), axis=1)
    df['longitude'] = df.apply(lambda row: convert_to_decimal(row['longitude'], row['long_dir']), axis=1)
    return df.sort_values(['date', 'time'])

def read_signature_information(path):
    """
    Read the signature information from a CSV file.
    :param path: str, path to the CSV file
    :return: pd.DataFrame
    """
    sig_info = pd.read_csv(path)
    sig_info['datetime'] = pd.to_datetime(sig_info['date'] + ' ' + sig_info['time'], format='%Y-%m-%d %H:%M:%S')
    sig_info['date'] = pd.to_datetime(sig_info['date'], format='%Y-%m-%d').apply(lambda value: value.date())
    sig_info['time'] = pd.to_datetime(sig_info['time'], format='%H:%M:%S').apply(lambda value: value.time())
    sig_info['sensor-time'] = pd.to_datetime(sig_info['sensor-time'])
    sig_info['gps-fix-time'] = pd.to_datetime(sig_info['gps-fix-time'])
    return sig_info


def read_target_run_log(path):
    """
    Read the target run log from a CSV file.
    :param path: str, path to the CSV file
    :return: pd.DataFrame
    """
    run_info = pd.read_csv(path)
    run_info['stop-datetime'] = pd.to_datetime(run_info['date'] + ' ' + run_info['stop-time-utc'], format='mixed', yearfirst=True)
    run_info['start-datetime'] = pd.to_datetime(run_info['date'] + ' ' + run_info['start-time-utc'], format='mixed', yearfirst=True)
    run_info['date'] = pd.to_datetime(run_info['date']).apply(lambda value: value.date())
    run_info['start-time-utc'] = pd.to_datetime(run_info['start-time-utc'], format='mixed').apply(lambda value: value.time()) 
    run_info['stop-time-utc'] = pd.to_datetime(run_info['stop-time-utc'], format='mixed').apply(lambda value: value.time()) 
    return run_info

def read_gps_log(path, start_date=None, end_date=None):
    """
    Read the GPS log from a CSV file.
    :param path: str, path to the CSV file
    :param start_date: datetime obj, start date and time
    :param end_date: datetime obj, end date and time
    :return: pd.DataFrame
    """
    gps_log = pd.read_csv(path)
    gps_log['datetime'] = pd.to_datetime(gps_log['date'] + ' ' + gps_log['time'], format='mixed', yearfirst=True)
    gps_log['date'] = pd.to_datetime(gps_log['date'], format='%Y-%m-%d')
    gps_log['time'] = pd.to_datetime(gps_log['time'], format='mixed')
    if start_date and end_date:
        gps_log = gps_log[(gps_log['datetime'] >= start_date) & (gps_log['datetime'] <= end_date)]
    elif start_date:
        gps_log = gps_log[gps_log['datetime'] >= start_date]
    elif end_date:
        gps_log = gps_log[gps_log['datetime'] <= end_date]
    return gps_log

def read_emplacement_information(path):
    """
    Read the emplacement log from a CSV file.
    :param path: str, path to the CSV file
    :return: pd.DataFrame
    """
    emplacement_log = pd.read_csv(path)
    emplacement_log['date'] = pd.to_datetime(emplacement_log['date'], format='%m/%d/%Y')
    return emplacement_log

def find_noise_times(sig_info, run_info):
    """
    Find the time ranges in which no vehicle is present.
    :param sig_info: pd.DataFrame, signature information
    :param run_info: pd.DataFrame, run information
    :return: pd.DataFrame
    """
    # Collect time ranges for all nodes
    subset = sig_info.drop_duplicates(subset=['location', 'date', 'node-id', 'mode'])
    all_gaps = []
    for _, row in subset.iterrows():
        _sig_info = sig_info[
            (sig_info['location'] == row['location']) &
            (sig_info['date'] == row['date']) &
            (sig_info['node-id'] == row['node-id']) &
            (sig_info['mode'] == row['mode'])
        ]
        sig_start = _sig_info['datetime'].min()
        sig_stop = _sig_info['datetime'].max()
        _run_info = run_info[
            (run_info['location'] == row['location']) &
            (run_info['date'] == row['date'])
        ].sort_values('start-datetime')
        start = _run_info['start-datetime'].iloc[0]
        stop = _run_info['stop-datetime'].iloc[0]
        intervals = []
        for _, run in _run_info.iterrows():
            intervals.append((start, stop))
            start = run['start-datetime']
            stop = run['stop-datetime']
        gaps = []
        previous_stop = sig_start
        for start, stop in intervals:
            if previous_stop < start:
                gaps.append((previous_stop, start))
            previous_stop = stop
        gaps.append((previous_stop, -1))
        gaps_df = pd.DataFrame(gaps, columns=['start', 'stop'])
        gaps_df['location'] = row['location']
        gaps_df['date'] = row['date']
        gaps_df['node-id'] = row['node-id']
        gaps_df['mode'] = row['mode']
        all_gaps.append(gaps_df)
    return pd.concat(all_gaps)        

def find_in_range_times(run_info, emplacement_info, delta, gps_log_path='data/position'):
    """
    Find the time range in which during a run the vehicle was within delta meters of an emplacement.
    :param run_info: pd.DataFrame, run information
    :param emplacement_info: pd.DataFrame, emplacement information
    :param delta: float, distance in meters
    :param gps_log_path: str, path to the GPS log files
    :return: pd.DataFrame
    """
    data = []
    header = ['run-idx', 'node-id', 'target-id', 'datetime-min', 'datetime-max', 'total-time', 'min-distance', 'max-distance']
    for run_idx, run in run_info.iterrows():
        placements = emplacement_info[emplacement_info['location'] == run['location']]
        run_placements = placements[placements['date'] == run['date']]
        gps_log_file = run['gps-log-file']
        gps_data = read_gps_log(os.path.join(gps_log_path, gps_log_file), start_date=run['start-datetime'], end_date=run['stop-datetime'])
        for node in run_placements['node-id'].unique():
            node_placement = run_placements[run_placements['node-id'] == node]
            gps_data['distance'] = gps_data.apply(lambda row: geodesic((row['latitude'], row['longitude']), (node_placement['lat'].iloc[0], node_placement['lon'].iloc[0])).meters, axis=1)
            in_range_times = gps_data[gps_data['distance'] <= delta]['datetime']
            _data = [
                run_idx,
                node,
                run['target-id'],
                in_range_times.min(),
                in_range_times.max(),
                in_range_times.max() - in_range_times.min(),
                gps_data['distance'].min(),
                gps_data['distance'].max(),
            ]
            data.append(_data)
    return pd.DataFrame(data, columns=header)