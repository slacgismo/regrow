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
    metadata = pd.read_csv("uspvdb.csv") 
    already_run = glob.glob(data_path +"/*.csv")
    already_run = [os.path.basename(x) for x in already_run]
    # Loop through the metadata and generate the associated estimates
    for idx, row in metadata.iterrows():
        lat = row['latitude']
        long = row['longitude']
        name = row['name']
        bus = row['bus']
        # Get the geohash associated with the site
        system_identifier = (bus + "_" + name + "_" +
                             str(lat) + "_" + str(long)).replace(" ", "_").replace("/", "_")
        if system_identifier + ".csv" in already_run:
            print("already run!!")
            continue
        geohash_val = geohash(lat, long, precision=6)
        # convert to KW
        power = row['capacity[MW]'] * 1000
        tilt = row['tilt[deg]']
        azimuth = row['azimuth[deg]']
        if row['axis'] == "FIXED_TILT":
            tracking = False
        else:
            tracking = True
        backtracking = False
        # Default tilt 0 for single-axis tracking if missing, 20 for fixed-tilt
        if str(tilt) == "nan":
            if tracking:
                tilt = 0
            else: 
                tilt = 20
        # Skip the system if it went online after 2022 
        if int(row['year']) > 2022:
            continue
        # Otherwise, start at 2018 or, if later, when the system came online
        if int(row['year']) > 2018:    
            min_measured_date = pd.to_datetime(str(int(row['year'])) + "-01-01 00:00:00")
        else:
            min_measured_date = pd.to_datetime("2018-01-01 00:00:00")
        max_measured_date = pd.to_datetime("2023-01-01 00:00:00")
        print(min_measured_date, max_measured_date)
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
                    break
                except:
                    pass
        # Build out the PVWatts model
        solpos = pvlib.solarposition.get_solarposition(master_weather_df.index,
                                                       lat, long)
        dni_extra = pvlib.irradiance.get_extra_radiation(master_weather_df.index)
        relative_airmass = pvlib.atmosphere.get_relative_airmass(solpos.zenith)
        temp_params = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
        pdc = run_pvwatts_model(tilt=tilt,
                                azimuth=azimuth,
                                dc_capacity=power,
                                dc_inverter_limit=power * 1.5,
                                solar_zenith=solpos.zenith,
                                solar_azimuth=solpos.azimuth, 
                                dni=master_weather_df['DNI'], 
                                dhi=master_weather_df['DHI'], 
                                ghi=master_weather_df['GHI'], 
                                dni_extra=dni_extra,
                                relative_airmass=relative_airmass, 
                                temperature=master_weather_df['Temperature'], 
                                wind_speed=master_weather_df['Wind Speed'],
                                temperature_model_parameters=temp_params,
                                temperature_coefficient=-0.0047,
                                tracking=tracking)
        # Plot PDC against real
        pdc.plot()
        # Plot real
        plt.show()
        plt.close()
        pdc.name = "output_kW"
        # Write the results to the associated S3 bucket.
        pdc.to_csv(os.path.join(data_path,
                                str(system_identifier) + ".csv"))
        
        
        
