"""
Aggregate powerplant-level predictions up to WECC node-level predictions.
Compare against rolled up WECC node-level predictions that didn't consider
power plant locations (simplified model).
"""

import pandas as pd
import matplotlib.pyplot as plt
import glob
from utils import geohash
import os

base_path = "C:/Users/kperry/Documents/extreme-weather-ca-heatwave"
power_plant_path = "pvwatts_powerplants"
wecc_node_path = "pvwatts_wecc_nodes"
aggregated_pp_wecc_node_path = "pvwatts_wecc_node_pp_agg"

powerplant_pvwatts_files = glob.glob(os.path.join(base_path,
                                                  power_plant_path, "*.csv"))
available_plant_geohashes = [os.path.basename(x).replace(".csv", "")
                             for x in powerplant_pvwatts_files]

wecc_pvwatts_files = glob.glob(os.path.join(base_path,
                                            wecc_node_path,
                                            "*.csv"))

metadata = pd.read_csv("C:/Users/kperry/Documents/source/repos/regrow/data/nodes_pvwatts_sim.csv")

unique_wecc_geocodes = list(metadata['geocode'].drop_duplicates())

for geocode in unique_wecc_geocodes:
    metadata_wecc_node = metadata[metadata['geocode'] == geocode]
    # Get a list of plants associated with the WECC node, open all of their
    # files and aggregate the associated PV production data
    associated_powerplants = metadata_wecc_node[[
        'plant_latitude', 'plant_longitude']].drop_duplicates()
    # Get all of the powerplant hashes associated with the WECC node
    powerplant_agg_list = list() 
    for idx, row in associated_powerplants.iterrows():
        plant_geohash = geohash(row['plant_latitude'],
                                row['plant_longitude'],
                                precision=6)
        if plant_geohash != '000000':
            powerplant_agg_list.append(plant_geohash)
    # Create a master dataframe to append onto
    plant_agg_df = pd.DataFrame()
    for plant_geohash in powerplant_agg_list:
        if plant_geohash in available_plant_geohashes:
            plant_preds = pd.read_csv(os.path.join(base_path, 
                                                   power_plant_path, 
                                                   plant_geohash + ".csv"),
                                      index_col=0, parse_dates=True)
            plant_preds = plant_preds.rename(columns = {"0": plant_geohash})
            plant_agg_df = pd.concat([plant_agg_df, plant_preds], axis=1)
            
    # Aggregate all of the rows into a single summed value
    plant_agg_df['summed_wecc_node_output'] = plant_agg_df.sum(axis=1, numeric_only=True)
    # Write the summed results to the associated aggregated WECC node file
    plant_agg_df.to_csv(os.path.join(base_path, aggregated_pp_wecc_node_path, 
                                     geocode + ".csv"))
    
    
# Compare the aggregated powerplant WECC node data to the simplified WECC
# node data

wecc_comparison_list = list()

for wecc_node_file in wecc_pvwatts_files:
    wecc_df = pd.read_csv(wecc_node_file, index_col=0, parse_dates=True)
    # Read in the aggregated WECC node
    agg_wecc_df = pd.read_csv(os.path.join(base_path,
                                           aggregated_pp_wecc_node_path,
                                           os.path.basename(wecc_node_file)),
                                           index_col=0, parse_dates=True)
    # Compare the outputs of the two dataframe against each other
    wecc_diff_mean = abs(agg_wecc_df['summed_wecc_node_output'] - wecc_df['0']).mean()
    wecc_diff_median = abs(agg_wecc_df['summed_wecc_node_output'] - wecc_df['0']).median()
    number_pp = len(agg_wecc_df.columns) - 1
    wecc_comparison_list.append({"wecc_node": os.path.basename(wecc_node_file).replace(".csv", ""),
                                 "number_associated_powerplants": number_pp,
                                 "mean_diff_simplified_v_agg": wecc_diff_mean,
                                 "median_diff_simplified_v_agg": wecc_diff_median})
    
wecc_comparison = pd.DataFrame(wecc_comparison_list)