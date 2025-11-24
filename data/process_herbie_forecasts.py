import boto3
from aws_access_cred_manager import aws_keys_and_tokens
import pvdrdb_tools
import pandas as pd
import glob 
import os


key="YOUR KEY"
secret="YOUR SECRET"
token="YOUR TOKEN"

def geohash(latitude, longitude, precision=6):
    """Encode a position given in float arguments latitude, longitude to
    a geohash which will have the character count precision.
    """
    from math import log10
    __base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
    __decodemap = { }
    for i in range(len(__base32)):
        __decodemap[__base32[i]] = i
    del i
    lat_interval, lon_interval = (-90.0, 90.0), (-180.0, 180.0)
    geohash = []
    bits = [ 16, 8, 4, 2, 1 ]
    bit = 0
    ch = 0
    even = True
    while len(geohash) < precision:
        if even:
            mid = (lon_interval[0] + lon_interval[1]) / 2
            if longitude > mid:
                ch |= bits[bit]
                lon_interval = (mid, lon_interval[1])
            else:
                lon_interval = (lon_interval[0], mid)
        else:
            mid = (lat_interval[0] + lat_interval[1]) / 2
            if latitude > mid:
                ch |= bits[bit]
                lat_interval = (mid, lat_interval[1])
            else:
                lat_interval = (lat_interval[0], mid)
        even = not even
        if bit < 4:
            bit += 1
        else:
            geohash += __base32[ch]
            bit = 0
            ch = 0
    return ''.join(geohash)

already_inserted_files = [os.path.basename(x).replace('.csv', "") for x in glob.glob("C:/Users/kperry/Documents/forecast_data_pivot/*.csv")]

s3 = boto3.client('s3',
                  aws_access_key_id=key,
                  aws_secret_access_key=secret,
                  aws_session_token=token)
bucket_name = "regrow"
paginator = s3.get_paginator('list_objects_v2')
pages = paginator.paginate(Bucket=bucket_name,
                           Prefix="Weather_data/herbie_forecasts/raw")

key_list = list()
for page in pages:
    for obj in page['Contents']:
        key_list.append(obj['Key'])
dates = pd.date_range("2018-01-01", "2022-12-31", freq="6H")

for date in dates:
    date =str(date).replace(" ", "_").replace(":",  "_")
    date_files = [key for key in key_list if date in key]
    if date in already_inserted_files:
        print("already inserted, skipping...")
        continue
    # read all of the date files out, append together, make master file
    # and then save
    master_df = pd.DataFrame()
    for date_file in date_files:
        response = s3.get_object(Bucket=bucket_name, Key=date_file)
        df = pd.read_csv(response['Body'])
        if "latitude" in list(df.columns):
            df = df.rename(columns={"latitude": "point_latitude",
                                    "longitude": "point_longitude",
                                    'time_horizon_hrs': 'forecast_horizon_hrs'})
        master_df = pd.concat([master_df, df], axis=0, ignore_index=True)
    master_df['tag_final'] = ["_".join(x.split(":")[:-1]) for x in list(master_df['tag'])]
    master_df.loc[master_df['tag_final'] == "", "tag_final"] = master_df['tag']
    master_df = master_df[master_df['tag_final'] != "TCDC"]
    # Tag renamer to line up GEFS with HRRR
    column_rename_dict = {"DPT": "_DPT_2 m above ground",
                          "TCDC": "_TCDC_entire atmosphere",
                          "TMP": "_TMP_surface",
                          "UGRD": "_UGRD_80 m above ground",
                          "VGRD": "_VGRD_80 m above ground"}
    master_df['tag_final'] = master_df['tag_final'].replace(column_rename_dict)
    pivot_table_df = master_df.pivot_table(index=['forecast_time', 'forecast_horizon_hrs',
                                                  'point_latitude', 'point_longitude'], 
                                            columns='tag_final', values='value',
                                            aggfunc='sum')
    pivot_table_df.to_csv("C:/Users/kperry/Documents/forecast_data_pivot/" + date + ".csv")
    print("inserted successfully!")
    
# Now loop through all of the nodes and aggregate up the forecast data
pivot_table_df = pd.read_csv("C:/Users/kperry/Documents/source/repos/regrow/data/nodes.csv")
pivot_table_df = pivot_table_df.rename(columns={"Lat": "point_latitude",
                                                "Long": "point_longitude"})
lat_lon_df = pivot_table_df[['point_latitude', 'point_longitude']].drop_duplicates()

files = glob.glob("C:/Users/kperry/Documents/forecast_data_pivot/*")

for idx, row in lat_lon_df.iterrows():
    latitude, longitude = row['point_latitude'], row['point_longitude']
    df_lat_lon_forecast = pd.DataFrame()
    for file in files:
        df_forecast = pd.read_csv(file)
        df_forecast = df_forecast[(df_forecast['point_latitude'] == latitude) &
                                  (df_forecast['point_longitude'] == longitude)]
        df_lat_lon_forecast = pd.concat([df_lat_lon_forecast, df_forecast])
    geohash_val = geohash(latitude, longitude)
    df_lat_lon_forecast.to_csv("C:/Users/kperry/Documents/forecast_aggregated/" + str(geohash_val) + "_forecast.csv",
                               index=False)
    
    