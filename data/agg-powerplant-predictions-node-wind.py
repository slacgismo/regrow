import pandas as pd
import glob
import os
import s3fs
os.environ["HOME"] = "C:/users/kperry"

from utils import geohash

aws_profile = "991404956194_regrow-developer"
base_path = r"C:/Users/kperry/OneDrive - NLR/Documents/source/repos/regrow/data/pysam_wecc_nodes"
power_plant_path = "regrow/pysam_wind_powerplants/single_turbine_power_timeseries/"
aggregated_pp_wecc_node_path = "pysam_wind_bus_agg"
geopanel_file_path = "pysam_geopanel.csv"
metadata_path = "uswtdb.csv"

col_name = "power[kW]"
metadata = pd.read_csv(metadata_path)
# Pull only CA sites
# pull_ca = True
# if pull_ca:
#     metadata = metadata[metadata["county"].str.contains("CA")]
unique_wecc_geocodes = list(metadata['bus'].drop_duplicates())

s3_fs = s3fs.S3FileSystem(anon=False, profile=aws_profile)
powerplant_files = s3_fs.glob("s3://" + power_plant_path + "*.csv")

already_run_files = glob.glob(os.path.join(base_path,  aggregated_pp_wecc_node_path + "/*.csv"))




for bus in unique_wecc_geocodes:
    metadata_wecc_node = metadata[metadata['bus'] == bus]
    metadata_wecc_turbines = [f'{lat}_{lon}.csv' for lat, lon in zip(metadata_wecc_node['latitude'],
                                                                      metadata_wecc_node['longitude'])]
    
    # Get a list of plants associated with the WECC node, open all of their
    # files and aggregate the associated power production data
    associated_pp_files = [x for x in powerplant_files if
                            os.path.basename(x) in metadata_wecc_turbines]
    file_insert_path = os.path.join(base_path, aggregated_pp_wecc_node_path,
                                      bus + ".csv")
    # if file_insert_path in already_run_files:
    #     print("Already run!")
    #     continue
    
    # Get the associated plant files based on the bus number at the
    # Create a master dataframe to append onto
    plant_agg_df = pd.DataFrame()
    for power_plant_path in associated_pp_files:
        path_basename = os.path.basename(power_plant_path)
        lat, lon = (path_basename.split("_")[-2],
                    path_basename.split("_")[-1].replace(".csv", ""))
        plant_preds = pd.read_csv(("s3://" + power_plant_path),
                                  storage_options={"profile": aws_profile},
                                  index_col=0, parse_dates=True)
        plant_preds = plant_preds.astype(float)
        plant_preds = plant_preds.rename(columns={col_name: f"{lat}_{lon}"})
        plant_preds = plant_preds[~plant_preds.index.duplicated(keep='first')]
        plant_agg_df = pd.concat([plant_agg_df, plant_preds], axis=1)

    # Aggregate all of the rows into a single summed value
    plant_agg_df["sum_pp"] = plant_agg_df.sum(axis=1, numeric_only=True)
    # Write the summed results to the associated aggregated WECC node file
    plant_agg_df.to_csv(file_insert_path)


# Generate a geopanel dataframe based on the data from all of the WECC nodes
aggregated_wecc_node_files = glob.glob(os.path.join(
    base_path, aggregated_pp_wecc_node_path, "*.csv"))
master_geopanel_df = pd.DataFrame()
for file in aggregated_wecc_node_files:
    wecc_node_value = os.path.basename(file).split(".")[0]
    df = pd.read_csv(file, index_col=0, parse_dates=True)
    bus = os.path.basename(file).replace(".csv", "")
    # Round output to 2 sigfigs
    df[bus] = df['sum_pp'].round(2)
    df.index = df.index.tz_convert('UTC')
    master_geopanel_df = pd.concat([master_geopanel_df, df[[bus]]], axis=1)


# Write the geopanel to a csv in REGROW /data/ folder
master_geopanel_df.to_csv(geopanel_file_path)