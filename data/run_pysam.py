import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
from utils import geohash, nsrdb_weather
import s3_creds
import glob as glob
import time
from requests.exceptions import HTTPError

import PySAM.Windpower as wp
import PySAM.ResourceTools as tools

def filter_uswtdb_metadata(uswtdb_df):
    """
    Replaces wind turbine model with the most common model for any model with
    missing parameteres needed to run PySAM wind model.
    """
    required_metadata = ["gentype", "capacity[MW]", "hub_height[m]",
                         "rotor_diameter[m]", "cut_in_wind_speed[m/s]",
                         "rated_wind_speed[m/s]", "cut_out_wind_speed[m/s]",
                         "max_rotor_speed[rd/min]", "drivetrain_design"]
    # Get most common wind turbine gentype
    common_tur_model = uswtdb_df.groupby(["gentype"]).size().idxmax()
    common_tur_metadata = uswtdb_df[uswtdb_df["gentype"]==common_tur_model]
    common_tur_metadata = common_tur_metadata[required_metadata]
    # Turbines with missing metadata gets replaced with common turbine metadata
    nan_rows = uswtdb_df[uswtdb_df.drop(columns=["drivetrain_design"]).isnull().any(axis=1)]
    nan_rows.loc[nan_rows.index, required_metadata] = common_tur_metadata.iloc[0].values
    uswtdb_df.update(nan_rows, overwrite=True)    
    
    start_year_list = []
    end_year_list = []
    # Filter out post 2022 production year and nan values
    uswtdb_df = uswtdb_df.loc[(uswtdb_df["year"]<=2022) &
                              (uswtdb_df["year"].notna())]
    # Get min and max model simulation dates
    for idx, row in uswtdb_df.iterrows():
        year = int(row["year"])
        # If pre2018 production year, start at 2018
        if year <= 2018: 
            start_year_list.append("1/1/2018")
            # If post2018 production year start at that post year
        else:
            start_year_list.append("1/1/" + str(year)) 
        end_year_list.append("12/30/2022")
        
    uswtdb_df["min_measured_date"] = start_year_list
    uswtdb_df["max_measured_date"] = end_year_list
    
    return uswtdb_df
    
 
def csv_to_srw(csv, site_id, site_year, site_lat, site_lon,
               site_elevation, hub_height, site_county):
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
    h1 = np.array([site_id, 'city??', site_county, 'USA', site_year,
                   site_lat, site_lon, site_elevation, 'tz???', 8760])  # meta info
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

    out = pd.concat([header, df], axis='rows')
    out.reset_index(drop=True, inplace=True)
    return out


def pull_nsrdb_data(min_measured_date, max_measured_date):
    """
    Pulls the nsrdb weather data between the min and max measured date
    in 30 mins interval.
    """
    all_nsrdb_data = pd.DataFrame()
    # Get nsdrdb data if not already gotten
    for year in range(min_measured_date.year, max_measured_date.year+1):
        try:
            # Pull the site's associated NSRDB data
            weather_df = nsrdb_weather(geohash_val,
                                        year,
                                        interval=30,
                                        attributes={'speed': 'wind_speed',  # m/s
                                                    "direction": 'wind_direction',  # deg
                                                    "temperature": 'temp_air',  # C
                                                    "pressure": "pressure"})  # mbar
            weather_df.reset_index(names=['datetime'], inplace=True)
            all_nsrdb_data = pd.concat(
                [all_nsrdb_data, weather_df], ignore_index=True)
        # If exceed nsrdb hourly pull rates, wait an hour for refresh
        except HTTPError as e:
            print("Exceeded API pull limit. Pulling NSRDB in an hour: ",e)
            time.sleep(3600)
        except Exception as e:
            print(e)
    # Save nsrdb pull
    all_nsrdb_data.to_csv(regrow_folder + "nsrdb/" + str(geohash_val) + "_nsrdb.csv",
        index=False,
        storage_options={"key": s3_creds.aws_aki,
                         "secret": s3_creds.aws_sak})
    return all_nsrdb_data



def run_single_turbine_pysam_model(rotor_diameter, hub_height, wind_speed,
                                   srw_file_path, cut_in_speed, cut_out_speed,
                                   shear_exponent, elevation,
                                   turbine_size, max_cp, max_tip_speed,
                                   max_tip_sp_ratio, drive_train, system_capacity,
                                   num_turbines, plot_powercurve=False):
    """
    Runs the PySAM model using NSRDB wind data across the time period
    and the common wind metadata for a single turbine.
    """
    # Create a new windpower turbine model
    wm = wp.new()
    
    # PySAM uses kW. Make MW to kW
    system_capacity = turbine_size
    turbine_size = turbine_size

    # Make a single wind turbine
    wm.value("wind_farm_wake_model", 0)
    wm.value('wind_resource_model_choice', 0)  # Hourly output
    # Generate a row of wind turbine coordinates sapaced 500m apart
    x_coords = [i*500 for i in range(num_turbines)]
    y_coords = [0] * num_turbines
    wm.value("wind_farm_xCoordinates", x_coords)
    wm.value("wind_farm_yCoordinates", y_coords)
    wm.value("system_capacity", system_capacity)
    # Calculate turbulence coefficient
    turbulence_coeff = (wind_speed.std()/wind_speed.mean()) * 100
    wm.value("wind_resource_turbulence_coeff", turbulence_coeff)
    
    # Get SRW data and put it in wind_resource_data
    data = tools.SRW_to_wind_data(srw_file_path)
    wm.value("wind_resource_data", data)

    # Set metadata parameters for model
    wm.value("wind_turbine_hub_ht", hub_height)
    wm.value("wind_resource_shear", shear_exponent)
    
    # Calculate power curve model with given metadata
    wm.Turbine.calculate_powercurve(
        turbine_size=turbine_size,
        rotor_diameter=rotor_diameter,
        elevation=elevation,
        max_cp=max_cp,
        max_tip_speed=max_tip_speed,
        max_tip_sp_ratio=max_tip_sp_ratio,
        cut_in=cut_in_speed,
        cut_out=cut_out_speed,
        drive_train=drive_train)  
    wind_sp = wm.Turbine.wind_turbine_powercurve_windspeeds
    power_out = wm.Turbine.wind_turbine_powercurve_powerout
    if plot_powercurve:
        plt.plot(wind_sp, power_out)
        plt.title("Power Curve")
        plt.ylabel("Power Output [kW]")
        plt.xlabel("Wind Speeds [m/s]")
        plt.show()

    # Run model
    wm.execute()

    # Get power output (System power generated [kW])
    power_output_kW = wm.Outputs.gen

    return power_output_kW


if __name__ == "__main__":
    # Point towards the particular local folder that contains the data
    data_path = "./pysam_wecc_nodes"
    # Get uswtdb metadata file
    uswtdb_df = pd.read_csv("uswtdb_metadata.csv")
    master_df = filter_uswtdb_metadata(uswtdb_df)
    master_df.to_csv("uswtdb_pysam_sim.csv", index=False)
    regrow_folder = "s3://pvdrdb-analysis/REGROW_Wind_Data/pysam_powerplants/"
    filename_list = [] 
    ran_geohash_nsrdb = []
    
    # Get a list of sites already ran
    ran_files_list = [file.replace(".png", "").split("\\")[-1]
                      for file in glob.glob("./pysam_wecc_nodes/plots/*.png")]
       
    for idx, row in master_df.iterrows():
        # Get metadata and make into correct dtype
        bus = row["bus"]
        # PySAM uses kW. Make MW to kW
        turbine_capacity = float(row["capacity[MW]"]) * 1000
        hub_height = float(row["hub_height[m]"])
        rotor_diameter = int(row["rotor_diameter[m]"])
        lat = row['latitude']
        lon = row['longitude']
        county = row["county"]
        elevation = float(row["elevation[m]"] + hub_height)
        cut_in = float(row["cut_in_wind_speed[m/s]"])
        rated_wind_spd = float(row["rated_wind_speed[m/s]"])
        cut_out = float(row["cut_out_wind_speed[m/s]"])
        max_rotor_spd = float(row["max_rotor_speed[rd/min]"])
        drivetrain_design = row["drivetrain_design"]
        min_measured_date = pd.to_datetime(row['min_measured_date'])
        max_measured_date = pd.to_datetime(row['max_measured_date'])
 
        max_cp = 0.47 #average max cp
        shear_exponent = 0.143 #average shear exponent
        
        # Calculate max tip speed and max tip speed ratio
        max_tip_spd = ((2*np.pi*max_rotor_spd) / 60) * (rotor_diameter/2)
        max_tip_spd_ratio = int(round(max_tip_spd/rated_wind_spd))
        
        # PySAM categorize the drive train into the following integer category 
        # 0: 3 Stage Planetary (most common)
        # 1: Single Stage - Low Speed Generator
        # 2: Multi-Generator
        # 3: Direct Drive
        if drivetrain_design == "Single Stage - Low Speed Generator":
            drivetrain_int = 1
        elif drivetrain_design == "Multi-Generator":
            drivetrain_int = 2
        elif drivetrain_design == "Direct Drive":
            drivetrain_int = 3
        else:
            drivetrain_int = 0
            
        # Get the geohash associated with the site
        geohash_val = geohash(lat, lon, precision=6)
        
        filename = f"{bus}_{lat}_{lon}"
        filename_list.append(filename)
        
        # If site is already ran, then skip
        if filename in ran_files_list:
            print("Modeled:", filename)
            pass
        elif filename =="9q713t_35.108196_-118.359886":
            pass
        else:
            print("Modeling:", filename)
            all_nsrdb_data = pull_nsrdb_data(min_measured_date, max_measured_date)
            all_power_output = pd.DataFrame()
                
            # Pysam takes exactly 8760 (hourly data for 365days) data points
            # for a wind model run
            limit = 8760
            for i in range(0, len(all_nsrdb_data), limit):
                end = min(i + limit, len(all_nsrdb_data))
                temp_df = all_nsrdb_data[i:end]
                
                if len(temp_df) == limit:
                    year = pd.to_datetime(temp_df["datetime"].iloc[1000]).year
                    srw_file_path = os.path.join(data_path, "srw_files",
                                                  str(geohash_val) + f"_{i}.srw")
                    temp_file_path = os.path.join(data_path, "srw_files",
                                                  str(geohash_val) + f"_{i}.csv")
                    temp_df.to_csv(temp_file_path, index=False)
    
                    # Convert csv to srw file to use in PySAM
                    srw = csv_to_srw(temp_file_path, str(geohash_val), year,
                                      lat, lon, elevation, hub_height,
                                      county)
                    srw.to_csv(srw_file_path, index=False, header=False)
    
                    wind_speed = temp_df["speed"]
                    # Run the PySAM model for one turbine
                    power_output = run_single_turbine_pysam_model(
                        wind_speed=wind_speed,
                        srw_file_path=srw_file_path,
                        rotor_diameter=rotor_diameter,
                        hub_height=hub_height,
                        cut_in_speed=cut_in,
                        cut_out_speed=cut_out,
                        shear_exponent=shear_exponent,
                        elevation=elevation,
                        turbine_size=turbine_capacity,
                        max_cp=max_cp,
                        max_tip_speed=max_tip_spd,
                        max_tip_sp_ratio=max_tip_spd_ratio, 
                        drive_train=drivetrain_int,
                        system_capacity=turbine_capacity,
                        num_turbines=1)
    
                    power_output_df = pd.DataFrame({
                        "datetime": pd.to_datetime(temp_df["datetime"]),
                        "power[kW]": power_output})
    
                    all_power_output = pd.concat(
                        [all_power_output, power_output_df], ignore_index=True)
                    
                    # Remove srw files when done
                    os.remove(temp_file_path)
                    os.remove(srw_file_path)
            
            # Plot power_output for all years
            plt.plot(all_power_output["datetime"],
                      all_power_output["power[kW]"])
            plt.title("Wind Predicted Power Output")
            plt.ylabel("Power Output [kW]")
            # Set datetime ticks to be only year
            plt.gca().xaxis.set_major_locator(mdates.YearLocator())
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            plt.xlabel("Year")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join("pysam_wecc_nodes", "plots", filename + ".png"))
            plt.show()
    
            # Save results
            all_power_output.to_csv((regrow_folder + filename + ".csv"),
                index=False,
                storage_options={"key": s3_creds.aws_aki,
                                  "secret": s3_creds.aws_sak})
        

     