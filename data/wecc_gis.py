import marimo as mo
import pandas as pd
from numpy import array
import os, sys, json

gis = pd.read_csv("wecc240_gis.csv")

# if not os.path.exists("powerplants.json") or os.path.getctime("powerplants.json") < os.path.getctime("powerplants.glm"):
#     os.system("gridlabd -C powerplants.glm -o powerplants.json")
# with open("powerplants.json","r") as _fh:
#     gen = dict([(x,y) for x,y in json.load(_fh)["objects"].items() if y['class'] == "powerplant"])

# _xx,_yy = array(gis.Lat),array(gis.Long)
# names = gis[gis.columns[0]]
# with open("powerplants_gis.glm","w") as _fh:
#     for _name,_plant in gen.items():
#         if not "latitude" in _plant:
#             print(f"// powerplant {_name} does not have a location",file=_fh)
#             continue
#         _x,_y = float(_plant["latitude"]),float(_plant["longitude"])
#         _dx,_dy = _xx-_x,_yy-_y
#         _d = zip(_dx*_dx + _dy*_dy,names)
#         _dn = sorted(_d,key=lambda x:x[0])
#         print("modify",f"{_name}.parent",f"wecc240_psse_N_{_dn[0][1]};",file=_fh)

#
# Write GLM gis data
#
with open("wecc240_gis.glm","w") as _fh:
    for _n,_col in gis.iterrows():
        name = f"wecc240_N_{_col.iloc[0]}"
        print(f"modify wecc240_psse_N_{_col.iloc[0]}.latitude {_col.Lat};",file=_fh)
        print(f"modify wecc240_psse_N_{_col.iloc[0]}.longitude {_col.Long};",file=_fh)
