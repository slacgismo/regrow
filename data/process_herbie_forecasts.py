import boto3
from aws_access_cred_manager import aws_keys_and_tokens
import pvdrdb_tools
import pandas as pd
import glob 
import os
from functools import reduce
import numpy as np

key="REGROW_KEY"
secret="REGORW_SECRET"
token="REGROW_TOKEN"


if __name__ == "__main__":
    
    ######## STEP 1: PULL ALL RAW FORECASTS AND FUSE THEM TOGETHER ############
    already_inserted_files = [os.path.basename(x).replace('.csv', "") for x in 
                              glob.glob("C:/Users/kperry/Documents/forecast_data_pivot/*.csv")]
    
    s3 = boto3.client('s3',
                      aws_access_key_id=key,
                      aws_secret_access_key=secret,
                      aws_session_token=token)
    bucket_name = "pvdrdb-transfer"
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name,
                               Prefix="REGROW/herbie_forecasts/raw")
    
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
        if len(master_df) > 0:
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
    
    ######## STEP 2: WIND PLANT-LEVEL FORECASTS #############################
    # Now loop through all of the wind plants and calculate
    # wind-plant level forecasts
    
    
    df = pd.read_csv("uswtdb.csv")
    df['Lat'] = df.groupby("name")['latitude'].transform('mean')
    df['Long'] = df.groupby("name")['longitude'].transform('mean')
    pivot_table_df = df[['name', 'Lat', "Long"]].drop_duplicates()
    
    pivot_table_df = pivot_table_df.rename(columns={"Lat": "point_latitude",
                                                    "Long": "point_longitude"})
    lat_lon_df = pivot_table_df[['name', 'point_latitude', 'point_longitude']].drop_duplicates()
    
    files = glob.glob("C:/Users/kperry/Documents/forecast_data_pivot/*")
    existing_files = glob.glob("C:/Users/kperry/Documents/forecast_aggregated/*.csv")
    existing_files = [os.path.basename(x) for x in existing_files]
    for idx, row in lat_lon_df.iterrows():
        name = row['name'].replace("/", " ")
        latitude, longitude = row['point_latitude'], row['point_longitude']
        if (name + "_" + str(latitude) + "_" +  str(longitude) + "_forecast.csv") in existing_files:
            print("Already inserted, skipping...")
            continue
        df_lat_lon_forecast = pd.DataFrame()
        for file in files:
            df_forecast = pd.read_csv(file)
            df_forecast = df_forecast[(df_forecast['point_latitude'] == latitude) &
                                      (df_forecast['point_longitude'] == longitude)]
            df_lat_lon_forecast = pd.concat([df_lat_lon_forecast, df_forecast])
        df_lat_lon_forecast.to_csv(("C:/Users/kperry/Documents/forecast_aggregated/" +
                                    name + "_" + str(latitude) + "_" + 
                                    str(longitude) + "_forecast.csv"),
                                   index=False)
    
    ######## STEP 3: WEIGHTED AGGREGATED TO NODE #############################
    # Aggregate all of the wind forecasts to the node level using
    # a weighted average
    
    df = pd.read_csv("uswtdb.csv")
    
    nodes = list(df['bus'].drop_duplicates())
    
    wind_forecasts = glob.glob("C:/Users/kperry/Documents/forecast_aggregated/*.csv")
    
    forecast_cols = ['_DPT_2 m above ground', '_PRES_surface',
                     '_RH_2 m above ground',
                     '_TCDC_entire atmosphere', '_TMP_surface', 
                     '_UGRD_80 m above ground',
                     '_VGRD_80 m above ground', 'point_latitude',
                     'point_longitude']
    
    for node in nodes:
        df_subset = df[df['bus'] == node].copy()
        df_subset['total_capacity[MW]'] = df_subset.groupby(
            ['name', 'bus'])['capacity[MW]'].transform('sum')
        # Take the earliest online year for each plant
        # just for simplification purposes
        df_subset['earliest_online_year'] = df_subset.groupby(
            ['name', 'bus'])['year'].transform('min')
        df_subset = df_subset[df_subset['year'] ==
                              df_subset['earliest_online_year']]
        # Get the name of the sites
        wind_sites = df_subset[['name', 'year',
                                'total_capacity[MW]']].copy().drop_duplicates()
        wind_sites = wind_sites[wind_sites['year']<=2022].copy()
        wind_sites['name'] = wind_sites['name'].str.replace("/", " ")
        # Loop thru sites
        forecast_df_list = list()
        for idx, row in wind_sites.iterrows():
            name = row['name']
            year = row['year']
            wind_forecast = [x for x in wind_forecasts
                             if name in x][0]
            forecast_df = pd.read_csv(wind_forecast)
            forecast_df['forecast_time'] = pd.to_datetime(
                forecast_df['forecast_time'], 
                format='mixed').dt.tz_localize("UTC")
            # Cutoff by the year it went online (also omit 2022
            # as it's missing random cases and wasn't full pulled)
            forecast_df = forecast_df[(forecast_df['forecast_time'].dt.year >= year) &
                                      (forecast_df['forecast_time'].dt.year < 2022)]
            # For each of the main columns, append the name of site
            for col in forecast_cols:
                forecast_df = forecast_df.rename(columns=
                                                 {col: col + "_" + name})
            forecast_df_list.append(forecast_df)
        # Merge all of the dataframes together
        df_merged = reduce(lambda a, b: a.merge(
            b, on=["forecast_time", "forecast_horizon_hrs"],
            how="outer"), forecast_df_list)
        # loop through each plant and add its capacity over
        # time as a column
        for idx, row in wind_sites.iterrows():
            df_merged[row['name']] = row['total_capacity[MW]']
            df_merged.loc[df_merged['forecast_time'].dt.year< row['year'],
                          row['name']] = np.nan
        # get the sum capacity over time across all plants
        df_merged['sum_capacity'] = df_merged[list(wind_sites[
           'name'].drop_duplicates())].sum(axis=1)
        # for each column type, calculate a weighted sum (use
        # sum capacity column and link to each site name)
        for col in forecast_cols:
            df_merged_cols = [x for x in df_merged.columns
                              if col in x]
            weighted_cols = list()
            for name in list(wind_sites['name'].drop_duplicates()):
                associated_col = [x for x in df_merged_cols if name in x][0]
                weight_score = df_merged[name] / df_merged['sum_capacity']
                df_merged[associated_col + "_weighted"] = df_merged[
                    associated_col] * weight_score
                # append the name of the weighted col to list
                # for final summed column
                weighted_cols.append(associated_col + "_weighted")
            # Sum all of the weighted columns together and then
            # get the final weighted value
            df_merged[col + "_weighted_avg"] = df_merged[
                weighted_cols].sum(axis=1)
        # Filter to only weighted columns
        final_col_list = (['forecast_time', 'forecast_horizon_hrs'] +
                          [x + "_weighted_avg" for x in forecast_cols])
        df_merged = df_merged[final_col_list]
        # Sort by forecast_time and forecast_horizon_hrs, and then
        # write to csv
        df_merged = df_merged.sort_values(by=['forecast_time', 
                                              'forecast_horizon_hrs'])
        # write the final weighted CSV file
        df_merged.to_csv(os.path.join("C:/Users/kperry/Documents/forecast_aggregated_node",
                                      node + "_weighted_forecast.csv"), index=False)
            
                    
