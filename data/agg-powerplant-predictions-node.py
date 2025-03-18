"""
Aggregate powerplant-level predictions up to WECC node-level predictions.
"""

import pandas as pd
import matplotlib.pyplot as plt
import glob
from utils import geohash
import os

base_path = "C:/Users/kperry/Documents/extreme-weather-ca-heatwave"
power_plant_path = "pvwatts_powerplants"
aggregated_pp_wecc_node_path = "pvwatts_bus_agg"
geopanel_file_path = "pvwatts_geopanel.csv"
metadata_path = "uspvdb.csv"
powerplant_files = glob.glob(os.path.join(base_path,
                                          power_plant_path, "*.csv"))

col_name = "output_kW"
metadata = pd.read_csv(metadata_path)

unique_wecc_geocodes = list(metadata['bus'].drop_duplicates())

for bus in unique_wecc_geocodes:
    metadata_wecc_node = metadata[metadata['bus'] == bus]
    # Get a list of plants associated with the WECC node, open all of their
    # files and aggregate the associated PV production data
    associated_pp_files = [x for x in powerplant_files if
                           os.path.basename(x).startswith(bus)]
    # Get the associated plant files based on the bus number at the
    # Create a master dataframe to append onto
    plant_agg_df = pd.DataFrame()
    for power_plant_path in associated_pp_files:
        path_basename = os.path.basename(power_plant_path)
        lat, lon = (path_basename.split("_")[-2], 
                    path_basename.split("_")[-1].replace(".csv", ""))
        plant_geohash = geohash(float(lat),
                                float(lon),
                                precision=6)
        plant_preds = pd.read_csv(power_plant_path,
                                  index_col=0, parse_dates=True)
        plant_preds = plant_preds.rename(columns = {col_name: plant_geohash})
        plant_agg_df = pd.concat([plant_agg_df, plant_preds], axis=1)
            
    # Aggregate all of the rows into a single summed value
    plant_agg_df["sum_pp"] = plant_agg_df.sum(axis=1, numeric_only=True)
    # Write the summed results to the associated aggregated WECC node file
    plant_agg_df.to_csv(os.path.join(base_path, aggregated_pp_wecc_node_path, 
                                     bus + ".csv"))
    

# Generate a geopanel dataframe based on the data from all of the WECC nodes

aggregated_wecc_node_files = glob.glob(os.path.join(base_path,
                                                    aggregated_pp_wecc_node_path, 
                                                    "*.csv"))
master_geopanel_df = pd.DataFrame()
for file in aggregated_wecc_node_files:
    wecc_node_value = os.path.basename(file).split(".")[0]
    df = pd.read_csv(file, index_col=0, parse_dates=True)
    master_geopanel_df = pd.concat([master_geopanel_df, df[[wecc_node_value]]], axis=1)
    
# Write the geopanel to a csv in REGROW /data/ folder
master_geopanel_df.to_csv(geopanel_file_path)