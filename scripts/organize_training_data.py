import os, sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from goodvibes.utils.data import organize_signatures

LOCATION_ROOTS = [
    '/home/chase/data/good_vibes/signature_data_files/bprf-sensor-data-ds2',
    '/home/chase/data/good_vibes/signature_data_files/eglin-sensor-data-ds2',
    '/home/chase/data/good_vibes/signature_data_files/cochise-sensor-data-ds2',
]

data = []
for location_root in LOCATION_ROOTS:
    data.append(organize_signatures(location_root))

stacked_data = pd.concat(data, ignore_index=True)

# Save data to CSV
stacked_data.to_csv('data/signature_information.csv', index=False)