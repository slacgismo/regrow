"""
Generate PVWatts models for each of the particular sites for the CA heatwave.
"""

import math
import numpy as np
import pvlib
import pandas as pd
import matplotlib.pyplot as plt 
import pathlib 
import requests
import json
import matplotlib
import time
import os

array_type_dict = {'Fixed - Open Rack': 0,
                   'Fixed - Roof Mounted': 1,
                   '1-Axis': 2,
                   '1-Axis Backtracking': 3,
                   '2-Axis': 4}

module_type_dict = {'Standard': 0,
                    'Premium': 1,
                    'Thin film': 2}

# Set regular losses at 14. this is changeable!
losses = 14


matplotlib.rcParams.update({'font.size': 12})

if __name__ == "__main__":
    # Point towards the particular local folder that contains the data  
    time_series_path = "C:/Users/kperry/Documents/extreme-weather-ca-heatwave"
    metadata = pd.read_csv(os.path.join(time_series_path,
                                        "all_states_heatwave_target_sites.csv"))
    for idx, row in metadata.iterrows():
        lat = row['latitude']
        long = row['longitude']
        power = row['power']
        tilt = row['tilt']
        azimuth = row['azimuth']
        min_measured_date = pd.to_datetime(row['min_measured_date'])
        max_measured_date = pd.to_datetime(row['max_measured_date'])
        tracking = row['tracking']
        backtracking = row['backtracking']
        mount_type = row['mount_type']
        module_type = row['module_type']
        # Set nan values as False
        if math.isnan(backtracking):
            backtracking = False
        if math.isnan(tracking):
            tracking = False
        # Get the array type
        if tracking and backtracking:
            array_type = 3
        elif tracking:
            array_type = 2
        elif 'roof' in mount_type.lower():
            array_type = 1
        else: 
            array_type = 0
        if module_type == 'CdTe':
            module_type = 2
        else:
            module_type = 0
        # Build out the payload to pass to the API
        payload = {'api_key': '4z5fRAXbGB3qldVVd3c6WH5CuhtY5mhgC2DyD952',
                   'system_capacity': power,
                   'module_type': module_type,
                   'losses': losses,
                   'array_type': array_type,
                   'tilt': tilt,
                   'azimuth': azimuth,
                   'lat': lat,
                   'lon': long,
                   'timeframe': 'hourly'}
        r = requests.get('https://developer.nrel.gov/api/pvwatts/v8.json?',
                         params = payload)
        model_outputs = json.loads(r.content.decode('utf-8'))
        hourly_outputs = model_outputs['outputs']['ac']