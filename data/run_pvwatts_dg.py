# -*- coding: utf-8 -*-
"""
Run PVWatts for DG
"""

from pvlib.modelchain import ModelChain
from pvlib.pvsystem import PVSystem
from pvlib.location import Location
import pandas as pd
import math
import requests
import json
import os
import pvlib
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
from matplotlib import pyplot as plt
import utils
import glob

def run_pvwatts_model(tilt, azimuth, dc_capacity, dc_inverter_limit,
                      solar_zenith, solar_azimuth, dni, dhi, ghi, dni_extra,
                      relative_airmass, temperature, wind_speed,
                      temperature_model_parameters,
                      temperature_coefficient, tracking):
    """
    Run the PVWatts model using NSRDB data across the time period as inputs.
    """
    if tracking:
        tracker_angles = pvlib.tracking.singleaxis(solar_zenith, solar_azimuth,
                                                   axis_tilt=tilt, axis_azimuth=azimuth,
                                                   backtrack=True, gcr=0.4, max_angle=60)
        surface_tilt = tracker_angles['surface_tilt']
        surface_azimuth = tracker_angles['surface_azimuth']
    else:
        surface_tilt = tilt
        surface_azimuth = azimuth
    
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt, surface_azimuth,
        solar_zenith,
        solar_azimuth,
        dni, ghi, dhi,
        dni_extra=dni_extra,
        airmass=relative_airmass,
        albedo=0.2,
        model='perez'
    )
    
    aoi = pvlib.irradiance.aoi(surface_tilt, surface_azimuth,
                               solar_zenith, solar_azimuth)
    # Run IAM model
    iam = pvlib.iam.physical(aoi, n=1.5)
    # Apply IAM to direct POA component only
    poa_transmitted = poa['poa_direct'] * iam + poa['poa_diffuse']
    temp_cell = pvlib.temperature.sapm_cell(
        poa['poa_global'],
        temperature,
        wind_speed,
        **temperature_model_parameters
    )
    pdc = pvlib.pvsystem.pvwatts_dc(
        poa_transmitted,
        temp_cell,
        dc_capacity,
        temperature_coefficient
    )
    return pdc


if __name__ == "__main__":
    # Point towards the particular local folder that contains the data
    metadata = pd.read_csv("wecc_bus_dg_cap_and_gen_by_month.csv")
    new_systems = list()
    email, api_key = utils.nsrdb_credentials()
    # Loop through the metadata and generate the associated estimates
    geohash_production_list = list()
    for geohash in list(metadata['geohash'].drop_duplicates()):
        print(f"Running monthly production for geohash {geohash}...")
        metadata_subset = metadata[metadata['geohash'] == geohash]
        lat = metadata_subset['lat'].iloc[0]
        long = metadata_subset['lon'].iloc[0]
        # Pull the site's associated NSRDB data 
        master_weather_df = pd.DataFrame()
        for year in range(2018, 2023):
            for try_time in range(0,3):
                try:
                    df, weather_metadata = pvlib.iotools.get_nsrdb_psm4_conus(latitude=lat,
                                                                              longitude=long,
                                                                              api_key=api_key,
                                                                              email=email,
                                                                              year=year,
                                                                              map_variables=True,
                                                                              time_step=30,
                                                                              )
                    master_weather_df = pd.concat([master_weather_df, df])
                    break
                except:
                    pass
        # print('weather df has duplicates:', master_weather_df.index.has_duplicates)
        # print('duplicated indices:', master_weather_df.index[master_weather_df.index.duplicated()].unique()[:5])
        # Loop through each month and generate the associated estimates for the
        # geohash
        agg_df_list = list()
        for idx, row in metadata_subset.iterrows():
            year = row['Year']
            month = row['Month']
            next_month = month % 12 + 1
            next_year = year + (1 if month == 12 else 0)
            power = row['Capacity [MW]'] * 1000
            lat = row['lat']
            long = row['lon']
            tilt = 20
            azimuth = 180
            tracking = False
            backtracking = False
            # Set the min and max date for the 
            min_measured_date = pd.to_datetime(f"{year}-{month}-01 00:00:00").tz_localize(master_weather_df.index.tz)
            max_measured_date = pd.to_datetime(f"{next_year}-{next_month}-01 00:00:00").tz_localize(master_weather_df.index.tz)
            weather_subset_df = master_weather_df[(master_weather_df.index >= min_measured_date) &
                                                  (master_weather_df.index < max_measured_date)]
            # Build out the PVWatts model
            solpos = pvlib.solarposition.get_solarposition(weather_subset_df.index,
                                                           lat, long)
            dni_extra = pvlib.irradiance.get_extra_radiation(weather_subset_df.index)
            relative_airmass = pvlib.atmosphere.get_relative_airmass(solpos.zenith)
            temp_params = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
            pdc = run_pvwatts_model(tilt=tilt,
                                    azimuth=azimuth,
                                    dc_capacity=power,
                                    dc_inverter_limit=power * 1.5,
                                    solar_zenith=solpos.zenith,
                                    solar_azimuth=solpos.azimuth, 
                                    dni=weather_subset_df['dni'], 
                                    dhi=weather_subset_df['dhi'], 
                                    ghi=weather_subset_df['ghi'], 
                                    dni_extra=dni_extra,
                                    relative_airmass=relative_airmass, 
                                    temperature=weather_subset_df['temp_air'], 
                                    wind_speed=weather_subset_df['wind_speed'],
                                    temperature_model_parameters=temp_params,
                                    temperature_coefficient=-0.0047,
                                    tracking=tracking)
            pdc.name = geohash
            agg_df_list.append(pd.DataFrame(pdc))
        node_production = pd.concat(agg_df_list)
        node_production = node_production[~node_production.index.duplicated(keep="first")].sort_index()
        geohash_production_list.append(node_production)
        geohash_output = pd.concat(geohash_production_list, axis=1)
        # Write it to a master CSV file
        geohash_output.to_csv("residential_solar_geopanel.csv")