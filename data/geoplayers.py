import os, sys
import pandas as pd
pd.options.display.max_columns = None
pd.options.display.width = 1024

buildings = pd.read_csv("buildings.csv",
    index_col = ["State","county","BuildingType"],
    )

# loadshapes = pd.read_csv("loadshapes.csv.zip",
#     index_col = ["county","building_type","datetime"],
#     )

print(buildings)
