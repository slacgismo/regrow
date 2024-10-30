# -*- coding: utf-8 -*-
"""
Main runner for cleaning up the data sets and landing the processed data
in S3.
"""

import pandas as pd
import pvdrdb_tools as pvd
import qa_routines as qar
import s3fs
import os
import matplotlib.pyplot as plt

pvq = pvd.pvdrdb_queries.PVDRDBQuery()
pvq.connectToDB()

meta_df = pd.read_csv("s3://pvdrdb-transfer/REGROW/california-heat-wave-solar-data/all_states_heatwave_target_sites.csv",
                      storage_options=pvq.aws)
meta_df = meta_df[meta_df['system_id']>5176].drop_duplicates()

for idx, row in meta_df.iterrows():
    sys_id = row['system_id']
    latitude = row['latitude']
    longitude = row['longitude']
    mount = "fixed"
    try:
        df = pd.read_csv("s3://pvdrdb-transfer/REGROW/california-heat-wave-solar-data/data/"
                         + str(sys_id) + ".csv",
                         storage_options=pvq.aws, index_col=0, parse_dates=True)
        # Get the associated PSM3 data for the site
        s3 = s3fs.S3FileSystem(anon=False, key = pvq.aws['key'],
                               secret = pvq.aws['secret'])
        nsrdb_files = s3.glob("s3://pvdrdb-inbox/Analysis_input/NSRDB/" + str(sys_id) +"_*.csv")
        psm3 = pd.DataFrame()
        for nsrdb_file in nsrdb_files:
            if 'tmy' not in nsrdb_file:
                psm3_sub = pd.read_csv(
                    "s3://" + nsrdb_file,
                    index_col=0, parse_dates=True,
                    storage_options=pvq.aws)
                psm3 = pd.concat([psm3, psm3_sub])
            # reorder the NSRDB data just in case
            psm3 = psm3.sort_index()
    except:
        continue
    master_df_filtered = pd.DataFrame()
    # loop through each column and run pre-processing on each column
    for col in list(df.columns):
        time_series = df[col].astype(float)
        try:
            if (("ac_power" in col) | ("dc_power" in col)):
                time_series = qar.run_power_stream_routine(time_series, latitude,
                                                           longitude, psm3, mount)
                # Append to dataframe
                if len(pd.Series(time_series.dropna().index.date).drop_duplicates()) > 365:
                    master_df_filtered[col] = time_series
            elif ("irradiance" in col):
                time_series = qar.run_irradiance_stream_routine(time_series,
                                                                latitude, longitude, psm3)
                # Append to dataframe
                if len(pd.Series(time_series.dropna().index.date).drop_duplicates()) > 365:
                    master_df_filtered[col] = time_series
            elif ("irradiance" in col):
                time_series = qar.run_irradiance_stream_routine(time_series,
                                                                latitude, longitude, psm3)
                # Append to dataframe
                if len(pd.Series(time_series.dropna().index.date).drop_duplicates()) > 365:
                    master_df_filtered[col] = time_series
            elif (("temp" in col) & ("module" in col)):
                time_series = qar.run_temperature_stream_routine(time_series, "module")
                # Append to dataframe
                if len(pd.Series(time_series.dropna().index.date).drop_duplicates()) > 365:
                    master_df_filtered[col] = time_series
            elif (("temp" in col) & ("ambient" in col)):
                time_series = qar.run_temperature_stream_routine(time_series, "ambient")
                # Append to dataframe
                if len(pd.Series(time_series.dropna().index.date).drop_duplicates()) > 365:
                    master_df_filtered[col] = time_series
            elif ("wind_speed" in col):        
                time_series = qar.run_wind_speed_stream_routine(time_series)
                # Append to dataframe
                if len(pd.Series(time_series.dropna().index.date).drop_duplicates()) > 365:
                    master_df_filtered[col] = time_series
        except Exception as e:
            print(e)
        if col in list(master_df_filtered.columns):
            time_series.plot()
            plt.title(col)
            plt.show()
            plt.close()
        # Write to a csv file
        master_df_filtered.to_csv(os.path.join(
            "C:/Users/kperry/Documents/regrow_pv_data_processed",
            str(sys_id) + ".csv"))
        