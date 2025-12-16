# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 16:48:40 2025

@author: kperry
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
from utils import geohash, nsrdb_weather
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
    data_path = "C:/Users/kperry/Documents/extreme-weather-ca-heatwave/pvwatts_powerplants"
    metadata = pd.read_csv("C:/Users/kperry/Downloads/wecc_bus_dg_cap_and_gen_by_month.csv") 
    already_run = glob.glob(data_path +"/*.csv")
    already_run_name = [os.path.basename(x).replace(".csv", "") for x in already_run]
    new_systems = list()
    # Loop through the metadata and generate the associated estimates
    for name in list(metadata['bus_name'].drop_duplicates()):
        subset = metadata[metadata['bus_name'] == name]
        row = subset.iloc[0]
        lat = row['lat']
        long = row['lon']
        bus = row['geohash']
        # Get the geohash associated with the site
        system_identifier = (bus + "_" + name + "_" +
                             str(lat) + "_" + str(long)).replace(" ", "_").replace("/", "_") +"_residential_solar"
        
        if system_identifier in already_run_name:
            print("already run!!")
            continue
        geohash_val = geohash(lat, long, precision=6)
        # convert to KW
        power = row['Capacity [MW]'] * 1000
        tilt = 20
        azimuth = 180
        tracking = False
        backtracking = False
        # Pull the site's associated NSRDB data 
        master_weather_df = pd.DataFrame()
        for year in range(2018, 2023):
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
                    break
                except:
                    pass
        # Build out the PVWatts model
        solpos = pvlib.solarposition.get_solarposition(master_weather_df.index,
                                                       lat, long)
        dni_extra = pvlib.irradiance.get_extra_radiation(master_weather_df.index)
        relative_airmass = pvlib.atmosphere.get_relative_airmass(solpos.zenith)
        temp_params = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
        # for each month calculate the expected pvwatts output (as nodes capacity is increasing on a monthly basis)
        pdc_aggregated_df = pd.DataFrame()
        for year in range(2018, 2023):
            for month in range(1, 13):
                subset_weather = master_weather_df[(master_weather_df.index.month == month) &
                                                   (master_weather_df.index.year == year)]
                pdc = run_pvwatts_model(tilt=tilt,
                                        azimuth=azimuth,
                                        dc_capacity=power,
                                        dc_inverter_limit=power * 1.5,
                                        solar_zenith=solpos.zenith,
                                        solar_azimuth=solpos.azimuth, 
                                        dni=subset_weather['DNI'], 
                                        dhi=subset_weather['DHI'], 
                                        ghi=subset_weather['GHI'], 
                                        dni_extra=dni_extra,
                                        relative_airmass=relative_airmass, 
                                        temperature=subset_weather['Temperature'], 
                                        wind_speed=subset_weather['Wind Speed'],
                                        temperature_model_parameters=temp_params,
                                        temperature_coefficient=-0.0047,
                                        tracking=tracking)
                pdc_aggregated_df = pd.concat([pdc_aggregated_df, pdc.dropna()])
        # Write the results to the associated S3 bucket.
        pdc_aggregated_df.plot()
        plt.show()
        plt.close()
        pdc_aggregated_df.to_csv(os.path.join(data_path,
                                str(system_identifier) + ".csv"))
        
        
        
