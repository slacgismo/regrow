"""
This script reads the wecc240_psse_gen.csv file and generates the buslist for
each type of generator in the WECC model, e.g., buslist_Solar.csv. Generation
with types not identified in gentypes.py are listed in buslist_Unknown.csv.

"""

import pandas as pd
from gentypes import gentypes

# read the generators list manually extracted from the wecc240_psse.raw file
data = pd.read_csv("wecc240_psse_gen.csv",quotechar="'",index_col="I")

# remove whitespaces for column names and string data fields
data.columns = [x.strip() for x in data.columns]
data.ID = [x.strip() for x in data.ID]

# lookup the generator types from gentypes.py
data["gen"] = [gentypes[x] if x in gentypes else "Unknown" for x in data.ID]

# output each generator type buslist file
for gen in data.gen.unique():
    data[data.gen==gen].to_csv(f"buslist_{gen}.csv")
