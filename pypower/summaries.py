"""This script produces various summaries of the model needed for the
study. All output is written the 'summaries' folder.

This script should be rerun anytime one of the model inputs or libraries
is changed. This can be done by running `make summaries`.
"""

import os
from ppmodel import PPModel
from wecc240 import wecc240
import pandas as pd

pd.options.display.width = None
pd.options.display.max_columns = None
pd.options.display.max_rows = None

os.makedirs("summaries",exist_ok=True)

def gis_2011(options) -> pd.DataFrame:
    """Get the GIS data for the 2011 model"""
    model = PPModel(case=wecc240)
    return model.get_data("gis").set_index("GEOHASH")

def gis_2020(options) -> pd.DataFrame:
    """Get the GIS data for 2020 model"""
    model = PPModel(case=wecc240(options=["SCHEDULING"]))
    return model.get_data("gis").set_index("GEOHASH")

# #
# # Get gis GEN and LOAD counts by GEOHASH
# #
def node_gencount(options) -> pd.DataFrame:
    model = PPModel(case=wecc240(options=options))
    return model.get_data("gis").set_index("GEOHASH").sort_index()["GEN"].dropna().rename({"GEN":"GENCOUNT"}).groupby("GEOHASH").sum().astype(int)

# 
# Network graphs
#
# model = PPModel(case=wecc240(options=["SCHEDULING"]))
# print(model.get_graph())

def bus_catalog(options) -> pd.DataFrame:
    """Generate the full bus catalog"""
    model = PPModel(case=wecc240(options=options))
    data = pd.merge(model.get_data("bus"),model.get_data("gis"),left_on="BUS_I",right_on="BUS_I")
    data = pd.merge(data,model.get_data("gen"),left_on="BUS_I",right_on="GEN_BUS",how="outer").drop("GEN_BUS",axis=1)
    data["BUS_I"] = data.BUS_I.astype(int)
    data["BUS_TYPE"] = [["NONE","PQ","PV","REF"][round(x)] for x in data.BUS_TYPE.astype(int)]
    data.set_index(["GEOHASH","BUS_I"],inplace=True)
    data = data[["NAME","BUS_TYPE","BASE_KV","PD","PMAX"]]
    data.columns = ["NAME","BUS_TYPE","VOLTAGE","LOAD","GENERATION"]
    data["GENOK"] = 0
    data.loc[data.BUS_TYPE=="PV","GENOK"] = 1
    return data

def bus_nogen(options) -> pd.DataFrame:
    """Generate the list of busses that cannot accept new generation"""
    data = bus_catalog(options)
    n_genbus = pd.DataFrame(data[["GENOK","LOAD"]].groupby("GEOHASH").sum()) # count of how many PV busses are there
    return data.reset_index().set_index("GEOHASH").loc[(n_genbus.GENOK==0)&(n_genbus.LOAD==0)]


if __name__ == "__main__":

    model_options = ["SCHEDULING"]
    for summary in [
            "gis_2011","gis_2020",
            "node_gencount",
            "bus_catalog","bus_nogen",
            ]:
        globals()[summary](options=model_options).to_csv(f"summaries/{summary}.csv")
