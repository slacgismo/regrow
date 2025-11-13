"""
This script reads the wecc240_psse_gen.csv file and generates the buslist for
each type of generator in the WECC model, e.g., buslist_Solar.csv. Generation
with types not identified in gentypes.py are listed in buslist_Unknown.csv.

"""

import sys
import pandas as pd
from gentypes import gentypes
sys.path.append("../data")
import utils

pd.options.display.width = None
pd.options.display.max_columns = None

# read the generators list manually extracted from the wecc240_psse.raw file
data = pd.read_csv("wecc240_psse_gen.csv",quotechar="'",index_col="I")

# remove whitespaces for column names and string data fields
data.columns = [x.strip() for x in data.columns]
data.ID = [x.strip() for x in data.ID]

# lookup the generator types from gentypes.py
data["GENTYPE"] = [gentypes[x] if x in gentypes else "Unknown" for x in data.ID]

# get GIS data and add geohash column as bus name
gis = pd.read_csv("../data/wecc240_gis.csv",index_col=0)
gis["bus"] = [utils.geohash(x,y) for x,y in zip(gis.Lat,gis.Long)]

# join GIS data
result = data.join(gis)
result.index.name = "BUS_I"
result.reset_index(inplace=True)
result.set_index("bus",inplace=True)

# output each generator type buslist file
for gentype in data.GENTYPE.unique():
    result[result.GENTYPE==gentype].to_csv(f"buslist_{gentype.lower()}.csv")

# generate bus gis
bus_index = gis.reset_index().set_index("bus")
bus_index.to_csv("buslist_index.csv",index=True,header=True)
