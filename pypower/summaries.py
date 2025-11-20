"""This script produces various summaries of the model needed for the
study. All output is written the 'summaries' folder.
"""

import os
from ppmodel import PPModel
from wecc240 import wecc240
import pandas as pd

pd.options.display.width = None
pd.options.display.max_columns = None
pd.options.display.max_rows = None

os.makedirs("summaries",exist_ok=True)

# #
# # Get gis GEN and LOAD counts by GEOHASH
# #
def node_gencount():
    model = PPModel(case=wecc240)
    model.get_data("gis").to_csv("tests/gis_2011m.csv",index=False)
    model = PPModel(case=wecc240(options=["SCHEDULING"]))
    model.get_data("gis").to_csv("tests/gis_2020m.csv",index=False)
    return model.get_data("gis").set_index("GEOHASH").sort_index()["GEN"].dropna().rename({"GEN":"GENCOUNT"}).groupby("GEOHASH").sum().astype(int)

# 
# Network graphs
#
# model = PPModel(case=wecc240(options=["SCHEDULING"]))
# print(model.get_graph())

#
# Bus catalog --> bus_catalog.csv
#
def bus_catalog():
    model = PPModel(case=wecc240(options=["SCHEDULING"]))
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

#
# No generation busses --> bus_nogen.csv
#
def bus_nogen():
    data = bus_catalog()
    n_genbus = pd.DataFrame(data[["GENOK","LOAD"]].groupby("GEOHASH").sum()) # count of how many PV busses are there
    return data.reset_index().set_index("GEOHASH").loc[(n_genbus.GENOK==0)&(n_genbus.LOAD==0)]


if __name__ == "__main__":

    for summary in ["node_gencount","bus_catalog","bus_nogen"]:
        globals()[summary]().to_csv(f"summaries/{summary}.csv")
