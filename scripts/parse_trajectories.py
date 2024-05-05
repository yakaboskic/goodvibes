import os, sys
import argparse
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from goodvibes.utils.data import parse_nmea_data

# TRAJECTORY_FILES = [
#     "/mnt/c/Users/c_yak/Downloads/retargetgpslog/L16F0023_DS2.TXT"
# ]

# for file in TRAJECTORY_FILES:
#     with open(file, 'r') as f:
#         raw_data = f.readlines()[1:]
#     df = parse_nmea_data(raw_data)
#     print(df.head())

def parse(args):
    with open(args.file, 'r') as f:
        raw_data = f.readlines()[1:]
    df = parse_nmea_data(raw_data)
    df.to_csv(args.output, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse NMEA data from a file')
    parser.add_argument('file', type=str, help='Path to the NMEA file')
    parser.add_argument('-o', '--output', type=str, help='Path to the output CSV file', default='output.csv')
    args = parser.parse_args()
    parse(args)