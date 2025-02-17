import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
from utils import geohash, nsrdb_weather
import PySAM.Windpower as wp
import PySAM.ResourceTools as tools

def generate_turbine_metadata(df):
    """
    Generates a typical wind turbine metadata. Also calculates the number of
    turbines needed to reach system capacity based on the provided capacity.
    Common land-based wind configuration found on page 27-28:
        https://www.nrel.gov/docs/fy22osti/81209.pdf
    """

    df["min_measured_date"] = "1/1/2018"
    df["max_measured_date"] = "12/30/2022"
    # Set parameters according to common land-based wind turbine metadatas
    df["turbine_rating_MW"] = 2.8
    df["rotor_diameter_m"] = 125
    df["hub_height_m"] = 90
    df["max_rotor_tip_speed"] = 80
    df["tip_speed_ratio"] = 8
    df["max_cp"] = 0.47
    df["drivetrain_design"] = "Geared"
    df["cut_in_wind_speed_m/s"] = 3
    df["cut_out_wind_speed_m/s"] = 25
    df["shear_exponent"] = 0.143
    df["altitude_m"] = 450

    # Generate the number of turbines needed to reach system capacity
    df["num_turbines"] = (df['aggregated_bus_fractional_capacity_MW'] /
                          df["turbine_rating_MW"]).astype(int)
    # Get the calculated capacity closest to the actual system's
    df["lower_capacity_MW"] = df["num_turbines"] * df["turbine_rating_MW"]
    df["upper_capacity_MW"] = (df["num_turbines"]+1) * df["turbine_rating_MW"]

    # Select the capacity closest to the actual capacity
    df["calculated_capacity_MW"] = np.where(
        (abs(df['aggregated_bus_fractional_capacity_MW'] - df["lower_capacity_MW"]) <=
            abs(df['aggregated_bus_fractional_capacity_MW'] - df["upper_capacity_MW"])),
        df["lower_capacity_MW"],
        df["upper_capacity_MW"])
    # If the upper capacity is closer, add final turbine numbers by 1
    df["num_turbines"] = np.where(
        (df["calculated_capacity_MW"] == df["upper_capacity_MW"]),
        df["num_turbines"]+1,
        df["num_turbines"])
    df = df.drop(columns=["lower_capacity_MW", "upper_capacity_MW"])

    df["capacity_diff_MW"] = df[ 'aggregated_bus_fractional_capacity_MW'] - \
        df["calculated_capacity_MW"]

    return df


def csv_to_srw(csv, site_id, site_year, site_lat, site_lon,
               site_elevation, hub_height, site_state, site_tz):
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

    out = pd.concat([header, df], axis='rows')
    out.reset_index(drop=True, inplace=True)
    return out


def run_single_turbine_pysam_model(rotor_diameter, hub_height, wind_speed,
                                   srw_file_path, cut_in_speed, cut_out_speed,
                                   shear_exponent, elevation,
                                   turbine_size, max_cp, max_tip_speed,
                                   max_tip_sp_ratio, system_capacity, num_turbines):
    """
    Runs the PySAM model using NSRDB wind data across the time period
    and the common wind metadata for a single turbine.
    """
    # Create a new windpower turbine model
    wm = wp.new()

    # PySAM uses kW. Make MW to kW
    system_capacity = system_capacity * 1000
    turbine_size = turbine_size * 1000

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
        drive_train=0)  # Geared drive train design is 0
    wind_sp = wm.Turbine.wind_turbine_powercurve_windspeeds
    power_out = wm.Turbine.wind_turbine_powercurve_powerout
    # plt.plot(wind_sp, power_out)
    # plt.title("Power Curve")
    # plt.ylabel("Power Output [kW]")
    # plt.xlabel("Wind Speeds [m/s]")
    # plt.show()

    # Run model
    wm.execute()

    # Get power output (System power generated [kW])
    power_output_kW = wm.Outputs.gen

    return power_output_kW


if __name__ == "__main__":
    # Point towards the particular local folder that contains the data
    # metadata_df = pd.read_csv("nodes_pysam_sim.csv")
    # master_df = generate_turbine_metadata(metadata_df)
    metadata = pd.read_csv("nodes_pysam_sim.csv")
    data_path = "C:/Users/kperry/Documents/extreme-weather-ca-heatwave/pysam_powerplants"
    # Identify type of capacity we want to aggregate on (plant level, wecc node
    # level)
    capacity_type = "wecc_node"
    if capacity_type == "wecc_node":
        metadata = metadata[['system_id', 'geocode', 'Bus  Number', 'state',
                             'tzoffset', 'Bus  Name', 'bus_latitude',
                             'bus_longitude', 
                             'aggregated_bus_fractional_capacity_MW', 
                             'min_measured_date',
                             'max_measured_date', 'turbine_rating_MW', 
                             'rotor_diameter_m',
                             'hub_height_m', 'max_rotor_tip_speed', 
                             'tip_speed_ratio', 'max_cp',
                             'drivetrain_design', 'cut_in_wind_speed_m/s', 
                             'cut_out_wind_speed_m/s',
                             'shear_exponent', 'altitude_m', 'num_turbines',
                             'calculated_capacity_MW', 'capacity_diff_MW']].drop_duplicates()
        metadata = metadata.rename(columns={
            "bus_latitude": "latitude", 
            "bus_longitude": "longitude",
            'aggregated_bus_fractional_capacity_MW': "power"})
        metadata = metadata.dropna(subset=['latitude', 'longitude'])
    else:
        metadata = metadata[['system_id', 'geocode', 'county', 
                             'state', 'tzoffset', 'plant_latitude',
                             'plant_longitude', 'plant_fractional_capacity_MW', 
                             'generator','min_measured_date',
                             'max_measured_date',
                             'plant_capacity_MW', 'turbine_rating_MW', 
                             'rotor_diameter_m',
                             'hub_height_m', 'max_rotor_tip_speed', 
                             'tip_speed_ratio', 'max_cp',
                             'drivetrain_design', 'cut_in_wind_speed_m/s', 
                             'cut_out_wind_speed_m/s',
                             'shear_exponent', 'altitude_m', 'num_turbines',
                             'calculated_capacity_MW', 'capacity_diff_MW']].drop_duplicates()
        metadata = metadata.rename(columns={
            "plant_latitude": "latitude", 
            "plant_longitude": "longitude",
            'plant_fractional_capacity_MW': "power"})
        metadata = metadata.dropna(subset=['latitude', 'longitude'])
    geocode_error_list = []
    ran_geocode_list = [] 
    for idx, row in metadata.iterrows():
        geocode = row['geocode']
        if geocode not in ran_geocode_list:
            ran_geocode_list.append(geocode)
            lat = row['latitude']
            long = row['longitude']
            state = row["state"]
            tz = row["tzoffset"]
            # Get the geohash associated with the site
            geohash_val = geohash(lat, long, precision=6)

            # Get metadata and set to appropriate dtype
            turbine_rating = row["turbine_rating_MW"]
            system_capacity = row['calculated_capacity_MW']
            rotor_diameter = row["rotor_diameter_m"]
            hub_height = row["hub_height_m"]
            max_cp = row["max_cp"]
            max_tip_speed = row["max_rotor_tip_speed"]
            max_tip_sp_ratio = row["tip_speed_ratio"]
            cut_in = row["cut_in_wind_speed_m/s"]
            cut_out = row["cut_out_wind_speed_m/s"]
            shear_exponent = row["shear_exponent"]
            altitude = row["altitude_m"]
            num_turbines = row["num_turbines"]
            min_measured_date = pd.to_datetime(row['min_measured_date'])
            max_measured_date = pd.to_datetime(row['max_measured_date'])
            all_nsrdb_data = pd.DataFrame()
            all_power_output = pd.DataFrame()

            for year in range(min_measured_date.year, max_measured_date.year+1):
                # Pull the site's associated NSRDB data
                weather_df = nsrdb_weather(geohash_val,
                                            year,
                                            interval=60,
                                            attributes={'speed': 'wind_speed',  # m/s
                                                        "direction": 'wind_direction',  # deg
                                                        "temperature": 'temp_air',  # C
                                                        "pressure": "pressure"})  # mbar
                weather_df.reset_index(names=['datetime'], inplace=True)
                all_nsrdb_data = pd.concat(
                    [all_nsrdb_data, weather_df], ignore_index=True)
            # Pysam takes exactly 8760 data points for a wind model run
            limit = 8760
            for i in range(0, len(all_nsrdb_data), limit):
                srw_file_path = os.path.join(data_path, "temp",
                                              str(geocode) + f"_{i}.srw")
                temp_file_path = os.path.join(data_path, "temp",
                                              str(geocode) + f"_{i}.csv")

                end = min(i + limit, len(all_nsrdb_data))
                temp_df = all_nsrdb_data[i:end]
                
                if len(temp_df) == limit:
                    temp_df.to_csv(temp_file_path, index=False)

                    year = pd.to_datetime(temp_df["datetime"].iloc[1000]).year
                    print(year)
                    # Convert csv to srw file to use in PySAM
                    srw = csv_to_srw(temp_file_path, geocode, year,
                                      lat, long, altitude, hub_height,
                                      state, tz)
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
                        elevation=altitude,
                        turbine_size=turbine_rating,
                        max_cp=max_cp,
                        max_tip_speed=max_tip_speed,
                        max_tip_sp_ratio=max_tip_sp_ratio,
                        system_capacity=system_capacity,
                        num_turbines=num_turbines)

                    power_output_df = pd.DataFrame({
                        "datetime": temp_df["datetime"],
                        "power_kW": power_output})

                    power_output_df["power_MW"] = power_output_df["power_kW"] / 1000
                    power_output_df = power_output_df.drop("power_kW", axis=1)

                    all_power_output = pd.concat(
                        [all_power_output, power_output_df], ignore_index=True)

                # Plot power_output for all years
                plt.plot(all_power_output["datetime"],
                          all_power_output["power_MW"])
                plt.title("Wind Predicted Power Output")
                plt.ylabel("Power Output [MW]")
                plt.xlabel("Date")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(data_path, "plots",
                                          str(geocode) + ".png"))
                plt.show()

            # Save results
            all_power_output.to_csv(os.path.join(data_path,
                                                  str(geocode) + ".csv"))
