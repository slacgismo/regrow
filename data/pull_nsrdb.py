"""
Pull down the associated NSRDB data for the WECC sites.
"""

import pandas as pd
import pvlib
import os
import time
from geocode import geohash

# Let NSRDB login be global variables
nsrdb_api_key = '4z5fRAXbGB3qldVVd3c6WH5CuhtY5mhgC2DyD952'
nsrdb_user_email = "kirsten.perry@nrel.gov"

if __name__ == "__main__":
    file = open("wecc240_gis.glm", "r")
    content = file.read()
    file.close()
    
    lines = content.split("\n")
    
    wecc_list_lat = list()
    wecc_list_long = list()
    
    for line in lines:
        # Get the name of the bus
        wecc_name = line.split("modify ")[-1].split(".longitude")[0].split(".latitude")[0]
        # get latitude-longitude coordinates for the particular bus
        try:
            latitude_val = float(line.split(".latitude ")[-1].replace(";", ""))
        except:
            latitude_val = None
        try:
            longitude_val = float(line.split(".longitude ")[-1].replace(";", ""))
        except:
            longitude_val = None
        if latitude_val:
            wecc_list_lat.append({"wecc_name": wecc_name,
                              "latitude": latitude_val})
        if longitude_val:
            wecc_list_long.append({"wecc_name": wecc_name,
                                   "longitude": longitude_val})
    
    wecc_df_lat = pd.DataFrame(wecc_list_lat)
    wecc_df_long = pd.DataFrame(wecc_list_long)
    
    wecc_df = pd.merge(wecc_df_lat, wecc_df_long, on ='wecc_name')
    # Now that we've assembled the WECC listing of lat-long coordinates, let's
    # pull the associated data from NSRDB and save it to the NSRDB folder
    for idx, row in wecc_df.iterrows():
        latitude = row['latitude']
        longitude = row['longitude']
        wecc_name = row['wecc_name']
        geohash_val = geohash(latitude, longitude, precision=6)
        years = [*range(2018,2023)]
        if not os.path.exists(os.path.join("./nsrdb", geohash_val + ".csv")):
            try:
                # Pull 5 minute data fields first
                psm3s = list()
                for year in years:
                    # Pull from API and save locally
                    psm3, _ = pvlib.iotools.get_psm3(latitude, longitude,
                                                    nsrdb_api_key,
                                                    nsrdb_user_email, year,
                                                    attributes=['dhi',
                                                                'dni',
                                                                'ghi',
                                                                'wind_direction',
                                                                'wind_speed',
                                                                'air_temperature',
                                                                'dew_point',
                                                                'relative_humidity',
                                                                'total_precipitable_water'],
                                                     map_variables=True,
                                                     interval=5,
                                                     leap_day=True,
                                                     timeout=60)
                    psm3s.append(psm3)
                if len(psm3s) > 0:
                    psm3_5min = pd.concat(psm3s)
                # # Pull hourly data fields next
                # psm3s = list()
                # for year in years:
                #     # Pull from API and save locally
                #     psm3, _ = pvlib.iotools.get_psm3(latitude, longitude,
                #                                     nsrdb_api_key,
                #                                     nsrdb_user_email, year,
                #                                     attributes=['air_temperature',
                #                                                 'dew_point',
                #                                                 'relative_humidity',
                #                                                 'total_precipitable_water'],
                #                                      map_variables=True,
                #                                      interval=30,
                #                                      leap_day=True,
                #                                      timeout=60)
                #     psm3s.append(psm3)
                # if len(psm3s) > 0:
                #     psm3_hourly = pd.concat(psm3s)
                # Now we're going to remove the date-related columns and concat the
                # two dataframes together
                cols_to_remove = ['Year', 'Month', 'Day', 'Hour', 'Minute']
                psm3_5min = psm3_5min.drop(columns=cols_to_remove)
                # psm3_hourly = psm3_hourly.drop(columns=cols_to_remove)
                psm3_5min.index = pd.to_datetime(psm3_5min.index)
                # psm3_hourly.index = pd.to_datetime(psm3_hourly.index)
                # psm3_master = pd.merge(psm3_5min, psm3_hourly, left_on=psm3_5min.index,
                #                       right_on=psm3_hourly.index, how='left')
                psm3_master = psm3_5min
                psm3_master = psm3_master.rename(columns={"key_0": "datetime"})
                psm3_master = psm3_master.round(3)
                # Set datetime index to ISO UTC format
                psm3_master.index = psm3_master.index.tz_convert('UTC')
                psm3_master.index = psm3_master.index.map(lambda x: x.isoformat())
                psm3_master.to_csv(os.path.join("./nsrdb", geohash_val + ".csv"))
                time.sleep(1)
            except:
                print("Couldn't fetch NSRDB for the following bus:" + row['wecc_name'])