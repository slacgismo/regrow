"""
API hook for pulling down air quality index parameters from the Open
Weather Map API.
"""

import requests
import pandas as pd
import os
import time
from datetime import datetime
import glob


# Define the API endpoint and your API key
api_key = "eb88788bd1e62c62176f05c6f87f16cc"

# Function to get weather data
def get_aqi(latitude, longitude, start, end, api_key):
    # Define the parameters for the request
    base_url = \
        ("http://api.openweathermap.org/data/2.5/air_pollution/history?lat=" + 
         str(latitude) + "&lon="+ str(longitude) + "&start=" + str(start) + 
         "&end=" + str(end) + "&appid=" + str(api_key))   
    # Make the request
    response = requests.get(base_url)
    # Check for valid response
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print("Error:", response.status_code)
        return None


def format_aqi_data(dataframe):
    """
    Format the raw dataframe so it's useable: convert from UNIX timestamps,
    parse out dictionaries.
    """
    

if __name__ == "__main__":
    # Point towards the particular local folder that contains the data  \\
    data_path = "C:/Users/kperry/Documents/extreme-weather-ca-heatwave"
    metadata = pd.read_csv(os.path.join(data_path,
                                        "all_states_heatwave_target_sites.csv")) 
    already_run = glob.glob(os.path.join(data_path, "air_quality_data/*"))
    already_run_systems = [int(os.path.basename(x).replace(".csv", "")) for x in already_run]
    for idx, row in metadata.iterrows():
        lat = row['latitude']
        long = row['longitude']
        if row['system_id'] in already_run_systems:
            print("system already analyzed, skipping...")
            continue
        else:
            min_measured_date = int(time.mktime(pd.to_datetime(row['min_measured_date']).timetuple()))
            max_measured_date =  int(time.mktime(pd.to_datetime(row['max_measured_date']).timetuple()))
            # Query the historic data for the site
            aqi_data = get_aqi(latitude=lat,
                               longitude=long, 
                               start=min_measured_date, 
                               end=max_measured_date, 
                               api_key=api_key)
            aqi_df = pd.DataFrame(aqi_data['list'])
            if len(aqi_df) > 0:
                master_aqi_df =  aqi_df['components'].apply(pd.Series)
                master_aqi_df['dt'] = aqi_df['dt']
                master_aqi_df['aqi'] = aqi_df['main'].apply(pd.Series)
                master_aqi_df['dt'] = [
                    datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                    for ts in list(master_aqi_df['dt'])]
                master_aqi_df = master_aqi_df[['dt', 'aqi', 'co', 'no', 'no2',
                                               'o3', 'so2', 'pm2_5', 'pm10', 'nh3']].drop_duplicates()
                master_aqi_df.to_csv(os.path.join(data_path, "air_quality_data",
                                                  str(row['system_id']) + ".csv"), index=False)
            print("Data successfully pulled and saved!")