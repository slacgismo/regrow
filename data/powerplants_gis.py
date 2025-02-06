import marimo as mo
import pandas as pd
from numpy import array
import os, sys, json

gis = pd.read_csv("wecc240_gis.csv")

if not os.path.exists("powerplants.json") or os.path.getctime("powerplants.json") < os.path.getctime("powerplants.glm"):
    os.system("gridlabd -C powerplants.glm -o powerplants.json")
with open("powerplants.json","r") as _fh:
    powerplants = dict([(x,y) for x,y in json.load(_fh)["objects"].items() if y['class'] == "powerplant"])

if not os.path.exists("wecc240_psse.json") or os.path.getctime("wecc240_psse.json") < os.path.getctime("../model/wecc240_psse.glm"):
    os.system("gridlabd -C ../model/wecc240_psse.glm -o wecc240_psse.json")
with open("wecc240_psse.json","r") as _fh:
    gens = dict([(x,y) for x,y in json.load(_fh)["objects"].items() if y['class'] == "gen"])
with open("wecc240_psse.json","r") as _fh:
    busses = dict([(x,y) for x,y in json.load(_fh)["objects"].items() if y['class'] == "bus"])

    _xx,_yy = array(gis.Lat),array(gis.Long)
names = gis[gis.columns[0]]
done = []
with open("powerplants_gis.glm","w") as _fh:
    for gen,data in gens.items():
        print(f"modify {gen}.status OUT_OF_SERVICE;",file=_fh);

    for _name,_plant in powerplants.items():
        if not "latitude" in _plant:
            print(f"// powerplant {_name} does not have a location",file=_fh)
            continue
        _x,_y = float(_plant["latitude"]),float(_plant["longitude"])
        _dx,_dy = _xx-_x,_yy-_y
        _d = zip(_dx*_dx + _dy*_dy,names)
        _dn = sorted(_d,key=lambda x:x[0])
        _gen = f"wecc240_psse_G_{_dn[0][1]}"
        _bus = busses[f"wecc240_psse_N_{_dn[0][1]}"]["bus_i"]
        if _gen not in done:
            print(f"""object pypower.gen {{
    name {_gen};
    bus {_bus};
    status IN_SERVICE;
}}""",file=_fh)
            done.append(_gen)
        print("modify",f"{_name}.parent",f"{_gen};",file=_fh)

