import os, sys
import pandas as pd
pd.options.display.max_columns=None
from config import state_list
from states import fips

#
# Load building loadshape data
#
# pumas = {
#     "Alameda" : "G0600010",
#     "Contra Costa" : "G0600130",
#     "Los Angeles" : "G0600370",
#     "Riverside" : "G0600650",
#     "San Bernadino" : "G0600710",
#     "San Diego" : "G0600730",
#     "San Francisco" : "G0600750",
#     "San Mateo" : "G0600810",
#     "Santa Clara" : "G0600850",
# }

# load = pd.read_csv("res_loadshapes.csv.zip")
# load["in.county"] = [pumas[x] for x in load.county]
# load.reset_index(inplace=True)
# load.set_index(["in.county","building_type"],inplace=True)
# load.sort_index(inplace=True)
# valid = load.index.get_level_values(0).unique()
# valid = list(pumas.values())

valid = [f"G{y}" for y in [fips(x) for x in state_list] if not y is None]

#
# Load building floorarea metadata
#
os.makedirs("res_loadshapes",exist_ok=True)

print("Downloading residential building metadata",end="...",file=sys.stderr,flush=True)
meta = pd.read_parquet("https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2021/resstock_amy2018_release_1/metadata/metadata.parquet")
print("done",file=sys.stderr)

print("Cleaning up metadata",end="...",file=sys.stderr,flush=True)
meta.set_index("in.county",inplace=True)
meta.drop([x for x in meta.columns if not x in ["in.geometry_building_type_recs","in.sqft"]],axis=1,inplace=True)
print(meta)
quit()

meta.drop(meta[~meta.index.str[0:4].isin(valid)].index,inplace=True)
print("done",file=sys.stderr)

print("Saving metadata",end="...",file=sys.stderr,flush=True)
meta.to_csv("res_metadata.csv.zip",index=False,header=True,compression="zip")
print("done",file=sys.stderr)

restype = {"Multi-Family with 2 - 4 Units" : "MFS",
           "Multi-Family with 5+ Units" : "MFL",
           "Single-Family Attached" : "SFA",
           "Single-Family Detached" : "SFD",
           "Mobile Home" : "MH",
          }

meta["building_type"] = [restype[x] for x in meta["in.geometry_building_type_recs"]]
meta.drop("in.geometry_building_type_recs",axis=1,inplace=True)

meta.reset_index(inplace=True)
meta.set_index(["in.county","building_type"],inplace=True)

meta.to_csv("res_loadshapes/floorareas.csv",index=True,header=True)
