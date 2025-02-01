import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
from utils import geohash, nsrdb_weather

import PySAM.Windpower as wp
import PySAM.ResourceTools as tools

def csv_to_srw(csv, site_id, site_state, site_year, site_lat, site_lon,
               site_elevation, site_tz):
    """
    Converts a csv file to srw format to be used as a resource file in 
    PySAM model.
    Specific SAM srw requirements starts on page 10:
        https://sam.nrel.gov/images/web_page_files/sam-help-2020-2-29-r2_weather_file_formats.pdf
    """
    df = pd.read_csv(csv)

    # Get header title
    header = pd.read_csv(csv, nrows=1, header=None).values
    
    # Convert mbar to atm 
    df['pressure'] = df['pressure'] / 1013.25
        
    # Create header lines 
    h1 = np.array([site_id, 'city??', site_state, 'USA', site_year,
                    site_lat, site_lon, site_elevation, site_tz, 8760])  # meta info
    h2 = np.array(["WTK .csv converted to .srw for SAM", None, None,
                   None, None, None, None, None, None, None])  # descriptive text
    h3 = np.array(['temperature', 'pressure', 'direction',
                   'speed', None, None, None, None, None, None])  # variables
    h4 = np.array(['C', 'atm', 'degrees', 'm/s', None,
                   None, None, None, None, None])  # units
    h5 = np.array([hub_height, hub_height, hub_height, hub_height, None, None,
                    None, None, None, None])  # hubheight
    header = pd.DataFrame(np.vstack([h1, h2, h3, h4, h5]))
    
    # Clean up
    df = df[['temperature', 'pressure', 'direction', 'speed']]
    df.columns = [0, 1, 2, 3]
    assert df.shape == (8760, 4)
    
    out = pd.concat([header, df], axis='rows')
    out.reset_index(drop=True, inplace=True)
    return out

def generate_wind_sys():
    """
    Generates a simulation of wind systems based on nodes and common wind
    configuration/metadata.
    Common land-based wind configuration found on page 27-28:
        https://www.nrel.gov/docs/fy22osti/81209.pdf
    """
    nodes_df = pd.read_csv("nodes.csv")
    
    nodes_df["min_measured_date"] = "1/1/2020"
    nodes_df["max_measured_date"] = "12/31/2020"
    # Set parameters according to common land-based wind turbine metadatas
    nodes_df["turbine_rating_kW"] = 2800
    nodes_df["rotor_diameter_m"] = 125
    nodes_df["hub_height_m"] = 90
    nodes_df["max_rotor_tip_speed"] = 80
    nodes_df["tip_speed_ratio"] = 8
    nodes_df["max_cp"] = 0.47
    nodes_df["drivetrain_design"] = "Geared"
    nodes_df["cut_in_wind_speed_m/s"] = 3 
    nodes_df["cut_out_wind_speed_m/s"] = 25
    nodes_df["shear_exponent"] = 0.143
    nodes_df["altitude_m"] = 450
    
    return nodes_df
    
def run_pysam_model(rotor_diameter, hub_height, wind_speed,
                    srw_file_path, cut_in_speed, cut_out_speed,
                    shear_exponent, elevation,
                    turbine_size, max_cp, max_tip_speed,
                    max_tip_sp_ratio):
    """
    Runs the PySAM model using NSRDB wind data across the time period
    and the common wind metadata.
    """
    # Create a new windpower turbine model
    wm = wp.new()
    
    # Make a single wind turbine
    wm.value("wind_farm_wake_model", 0)
    wm.value("wind_farm_xCoordinates", [0])
    wm.value("wind_farm_yCoordinates", [0])
    wm.value("system_capacity", turbine_size)
    # Calculate turbulence coefficient
    turbulence_coeff = (wind_speed.std()/wind_speed.mean()) * 100
    wm.value("wind_resource_turbulence_coeff", turbulence_coeff)
    
    # Get SRW data and put it in wind_resource_data
    data = tools.SRW_to_wind_data(srw_file_path)
    wm.value("wind_resource_data", data)
    
    # Set metadata parameters for model
    wm.value("wind_turbine_rotor_diameter", rotor_diameter)    
    wm.value("wind_turbine_hub_ht", hub_height)
    wm.value("wind_resource_shear", shear_exponent)
    wm.value('wind_resource_model_choice', 0) # Hourly output
    
    # Calculate power curve model with given metadata
    wm.Turbine.calculate_powercurve(
        turbine_size = turbine_size,
        rotor_diameter= rotor_diameter,
        elevation = elevation,
        max_cp = max_cp,
        max_tip_speed = max_tip_speed,
        max_tip_sp_ratio = max_tip_sp_ratio,
        cut_in = cut_in_speed,
        cut_out = cut_out_speed,
        drive_train = 0) # Geared drive train design is 0
    wind_sp = wm.Turbine.wind_turbine_powercurve_windspeeds
    power_out = wm.Turbine.wind_turbine_powercurve_powerout
    # plt.plot(wind_sp, power_out)
    # plt.title("Power Curve")
    # plt.ylabel("Power Output [kW]")
    # plt.xlabel("Wind Speeds [m/s]")
    # plt.show()
    
    #Run model
    wm.execute(1)
    
    # Get power output (System power generated [kW])
    power_output = wm.Outputs.gen
    
    return power_output

if __name__ == "__main__":
    # Point towards the particular local folder that contains the data
    data_path = "./pysam_wecc_nodes"
    metadata = generate_wind_sys()
    geocode_error_list = []
    for idx, row in metadata.iterrows():
        
        lat = row['Lat']
        long = row['Long']
        geocode= row['geocode']
        state = row["state"]
        print(geocode)
        tz = row["tzoffset"]
        # Get the geohash associated with the site
        geohash_val = geohash(lat, long, precision=6)
        
        # Get metadata and set to appropriate dtype
        turbine_rating = row['turbine_rating_kW']
        rotor_diameter = row["rotor_diameter_m"]
        hub_height = row["hub_height_m"]
        max_cp = row["max_cp"]
        max_tip_speed = row["max_rotor_tip_speed"]
        max_tip_sp_ratio = row["tip_speed_ratio"]
        cut_in = row["cut_in_wind_speed_m/s"]
        cut_out = row["cut_out_wind_speed_m/s"]
        shear_exponent = row["shear_exponent"]
        altitude = row["altitude_m"]

        min_measured_date = pd.to_datetime(row['min_measured_date'])
        max_measured_date = pd.to_datetime(row['max_measured_date'])
        
        nsrdb_file_path = os.path.join(data_path,
            str(geocode) + "_nsrdb.csv")  
        srw_file_path = os.path.join(data_path,
            str(geocode) + "_nsrdb.srw")
        
        # Pull the site's associated NSRDB data
        weather_df = nsrdb_weather(geohash_val,
                            2020,
                            interval=60,
                            attributes={'speed': 'wind_speed', # m/s
                                        "direction": 'wind_direction', # deg
                                        "temperature": 'temp_air', #C
                                        "pressure":"pressure"}) #mbar
        
        weather_df.reset_index(names=['datetime'], inplace=True)
        # Get only the first 8760 rows since PySAM data limit is 8760
        weather_df = weather_df.head(8760)
        weather_df.to_csv(nsrdb_file_path, index=False)
        
        # Convert csv to srw file to use in PySAM 
        srw = csv_to_srw(nsrdb_file_path, geocode, state, 2020,
                                 lat, long, altitude, tz)
        srw.to_csv(srw_file_path,index=False, header=False)
        
        # Build out the PySAM model
        power_output = run_pysam_model(
            wind_speed=weather_df["speed"],
            srw_file_path=srw_file_path,
            rotor_diameter=rotor_diameter,
            hub_height=hub_height,
            cut_in_speed=cut_in,
            cut_out_speed=cut_out,
            shear_exponent=shear_exponent,
            elevation = altitude,
            turbine_size=turbine_rating,
            max_cp=max_cp,
            max_tip_speed=max_tip_speed,
            max_tip_sp_ratio=max_tip_sp_ratio)
        
        power_output_df = pd.DataFrame({
            "datetime": weather_df["datetime"],
            "power_kW": power_output})
        
        # Plot power_output
        plt.plot(weather_df["datetime"], power_output_df["power_kW"])
        plt.title("Predicted Power Output")
        plt.ylabel("Power Output [kW]")
        plt.xlabel("Date")
        plt.savefig(os.path.join(data_path, "plots", str(geocode) + ".png"))
        plt.show()
        
        # Write the results to the associated S3 bucket.
        power_output_df.to_csv(os.path.join(data_path,
            str(row['geocode']) + ".csv"))
            
            
            
