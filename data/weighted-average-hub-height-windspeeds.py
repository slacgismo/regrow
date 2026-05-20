import pandas as pd
import glob
import os
import s3fs
import requests
os.environ["HOME"] = "C:/users/kperry"
import boto3
import time
from utils import geohash
from utils import geohash, nsrdb_weather, nsrdb_credentials
import io

aws_profile = "991404956194_regrow-developer"
base_path = r"./pysam_wecc_nodes"
power_plant_path = "regrow/pysam_wind_powerplants/single_turbine_weather_data/"
aggregated_pp_wecc_node_path = "pysam_wind_bus_agg"
geopanel_file_path = "pysam_geopanel.csv"
metadata_path = "uswtdb.csv"


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
    weather_data = pd.read_csv(io.StringIO(x.content.decode('utf-8')), header=1)
    return weather_data


if __name__ == "__main__":
    # Build out SD3 credentials
    session = boto3.Session(profile_name=aws_profile)
    credentials = session.get_credentials()
    
    storage_options = {
        "key": credentials.access_key,
        "secret":  credentials.secret_key,
        "token": credentials.token
    }
    # Connect to client
    s3 = boto3.client('s3',
                      aws_access_key_id=storage_options['key'],
                      aws_secret_access_key=storage_options['secret'],
                      aws_session_token=storage_options['token'])
    
    s3_fs = s3fs.S3FileSystem(anon=False, profile=aws_profile)
    
    
    wind_df = pd.read_csv("uswtdb_pysam_sim.csv")
    node_df = pd.read_csv("nodes.csv")
    node_df = node_df.rename(columns={"Lat": "node_latitude",
                                      "Long": "node_longitude",
                                      "geocode": "bus"})
    node_df = node_df[['bus', 'node_latitude', 'node_longitude']]
    wind_df = pd.merge(wind_df, node_df, on='bus')
    nodes = list(wind_df['bus'].drop_duplicates())
    
    # Loop through each node and generate the turbine-averaged wind data
    for node in nodes:
        # get all turbines associated with a node
        wind_subset = wind_df[wind_df['bus'] == node]
        node_lat, node_lon = (wind_subset['node_latitude'].iloc[0],
                              wind_subset['node_longitude'].iloc[0])
        # Loop through all of the turbines and get the mean weighted wind speed, air pressure, air temperature, and wind direction
        wind_params_list = list()
        wind_params = ["Air Temperature", "Surface Air Pressure",  "Wind Speed",
                       "Wind Direction"]
        col_list = list()
        for idx, row in wind_subset.iterrows():
            lat, lon = row['latitude'], row['longitude']
            df = pd.read_csv(f"s3://regrow/pysam_wind_powerplants/single_turbine_weather_data/{lat}_{lon}.csv",
                           storage_options=storage_options)
            # get the datetime
            df['datetime'] = pd.to_datetime(
                df['Month'].astype(str) + "/" + 
                df['Day'].astype(str) + "/" + 
                df['Year'].astype(str) + " " +
                df['Hour'].astype(str) + ":" + 
                df['Minute'].astype(str) + ":00")
            df['datetime'] = df['datetime'].dt.tz_localize("UTC")
            df.index = df['datetime']
            # Select all of the columns associated with the params we wont (omit all those extra data/time columns
            # which are now handled via the datetime column)
            for param in wind_params:
                select_col = [x for x in list(df.columns) if param in x][0]
                df = df.rename(columns = {select_col: f"{select_col}_{lat}_{lon}"})
                col_list.append(df[f"{select_col}_{lat}_{lon}"])
        # Wind parameters 
        col_merged = pd.concat(col_list,axis=1)
        # get an averaged value of each with parameter
        for param in wind_params:
            columns = [x for x in col_merged.columns if param in x]
            col_merged[f'average_{select_col}'] = col_merged[columns].mean(axis=1)
            col_merged_subset = col_merged[[f'average_{select_col}']]
            wind_params_list.append(col_merged_subset)
        # Create a master dataframe of all of the averaged columns
        wind_merged_df = pd.concat(wind_params_list, axis=1)
        # write the merged wind speed to a dataframe
        wind_merged_df.to_csv(f"./pysam_wecc_nodes/weight_avg_node_wind/weighted_average_wind_node_{node}.csv")
        
        
            