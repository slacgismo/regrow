# -*- coding: utf-8 -*-
"""
Pull NSRDB and CONUS data at different nodes for comparison.
"""

import pandas as pd
from utils import geohash, nsrdb_weather, nsrdb_credentials
import requests
import io
import pvdrdb_tools
import time

pdv = pvdrdb_tools.PVDRDBQuery()

def pull_CONUS_data(latitude, longitude, hub_height,
                    year):
    available_hub_heights = [10] +[*range(20,200, 20)]
    # Get the closest hub height to the existing turbine
    closest_hub_height = int(min(available_hub_heights, 
                                 key=lambda x:abs(x-hub_height)))
    payload_attributes = ("temperature_" + str(int(closest_hub_height))+ 
                          "m,pressure_0m,windspeed_" + 
                          str(int(closest_hub_height))+ "m,winddirection_" +
                          str(int(closest_hub_height))+ "m")
    email, api_key = nsrdb_credentials()
    url = (
        "http://developer.nrel.gov/api/wind-toolkit/v2/"+
        "wind/wtk-bchrrr-v1-0-0-download.csv?wkt=POINT("
           + str(longitude) + " " + str(latitude) +
           ")&attributes=" + payload_attributes +
           "&names=" +str(year) +
           "&utc=true&leap_day=true&email=" + str(email) + "&api_key=" + 
           str(api_key))
    x = requests.get(url)
    # Process the data into CSV format
    weather_data = pd.read_csv(io.StringIO(x.content.decode('utf-8')))
    weather_data.drop(columns=weather_data.columns[-1], inplace=True)
    timezone = int(weather_data.columns[3])
    weather_data.columns = ['Year', 'Month', 'Day', 'Hour', 
                            'Minute', 'Air Temperature at hub height (°C)',
                            'Surface Air Pressure (Pa)',
                            'Wind Speed at hub height (m/s)',
                            'Wind Direction at hub height(°)']
    weather_data['hub_height'] = hub_height
    index_cutoff = weather_data[weather_data['Year'] =='Year'].index[0]
    weather_data = weather_data[weather_data.index > index_cutoff]
    weather_data.index = pd.to_datetime(weather_data['Year'] + "-" +
                                        weather_data['Month'] + "-" + 
                                        weather_data['Day'] + " " + 
                                        weather_data['Hour'] + 
                                        ":" + weather_data['Minute'] + ":00")
    weather_data.index = weather_data.index.tz_localize('UTC')
    weather_data = weather_data[['Air Temperature at hub height (°C)',
                                 'Surface Air Pressure (Pa)',
                                 'Wind Speed at hub height (m/s)',
                                 'Wind Direction at hub height(°)']]
    return weather_data


if __name__ == "__main__":
    # Point towards the particular local folder that contains the data
    metadata = pd.read_csv("nodes.csv")
    # Loop through the metadata and generate the associated estimates
    #metadata = metadata[metadata.index>=78]
    for idx, row in metadata.iterrows():
        lat = row['Lat']
        long = row['Long']
        geohash_val = geohash(lat, long, precision=6)
        min_measured_date = pd.to_datetime("2018-01-01")
        max_measured_date = pd.to_datetime("2022-12-31")
        # Pull the site's associated NSRDB data 
        master_weather_df = pd.DataFrame()
        for year in range(min_measured_date.year, max_measured_date.year):
            for try_time in range(0,3):
                try:
                    df = nsrdb_weather(geohash_val,
                                           year,
                                           interval=30,
                                           attributes={'Temperature': 'temp_air',
                                                       'DHI': 'dhi',
                                                       'DNI': 'dni',
                                                       'GHI': 'ghi',
                                                       'Wind Speed': 'wind_speed'})
                    master_weather_df = pd.concat([master_weather_df, df])
                except:
                    pass
        # Pull the CONUS data for the same associated period for direct comparison
        master_conus_df = pd.DataFrame()
        for year in range(min_measured_date.year, max_measured_date.year):
            tries = 0
            while tries < 3:
                try:
                    conus_df = pull_CONUS_data(lat, long, 80, year)
                    master_conus_df = pd.concat([master_conus_df, conus_df])
                    time.sleep(1)
                    break
                except:
                    tries += 1
                    time.sleep(1)
        # Merge the data sets together
        df_merge = pd.merge(master_weather_df, master_conus_df,
                            left_index=True, right_index=True)
        # Write to S3
        df_merge.to_csv('s3://pvdrdb-transfer/REGROW/nsrdb_conus_data/' 
                        + str(row['geocode']) + '.csv',
                        storage_options={"key": pdv.aws['key'],
                                         "secret": pdv.aws['secret']})