"""Split and reaggregate powerplants by WECC node

The script read the WECC240 model and splits up powerplant objects based on
the generator type. The operating capacities are summed and their
contribution to the total generating capacity is totaled.

Input file is `wecc240.json`. Intermediate file is `powerplants_split.csv`.
Output file is `powerplants_aggregated.csv`.

Output fields are as follows:

* `bus`: the geocode of the bus at which the generator type is aggregated

* `gen`: the generator type being aggregated

* `cap`: the capacity of the generation type (MW)

* `cf`: the contribution factor of the generation to the total bus generation capacity

* `units`: the number of units identified of this type at this bus

"""
import os
import sys
import json
sys.path.append("../data")
import utils
import pandas as pd

if not os.path.exists("powerplants_aggregated.csv"):

    data = json.load(open("wecc240.json","r"))
    assert(data["application"]=="gridlabd")
    objects = data["objects"]
    with open("powerplants_split.csv","w") as fh:
        print("name,bus,gen,cap,cf,units",file=fh)
        for name,properties in [(x,y) for x,y in objects.items() if y["class"] == "powerplant"]:
            try:
                bus = properties["parent"].replace("_N_","_G_")
                gen = properties["generator"]
                cap = float(properties["operating_capacity"].split()[0])
                pmax = float(objects[bus]["Pmax"].split()[0])
                node = objects[bus]["parent"]
                lat = float(objects[node]["latitude"])
                lon = float(objects[node]["longitude"])
                geo = utils.geohash(lat,lon)
                for item in gen.split("|"):
                    print(name,geo,item,cap,cap/pmax if pmax>0 else float('nan'),1,sep=",",file=fh)
            except:
                e_type,e_value,e_trace = sys.exc_info()
                print(f"ERROR [{name}]: {e_type.__name__} {e_value}",file=sys.stderr)

    data = pd.read_csv("powerplants_split.csv",index_col=["bus","gen"],usecols=["bus","gen","cap","cf","units"])
    data.groupby(["bus","gen"]).sum().round(3).to_csv("powerplants_aggregated.csv")

    print(data.groupby(["bus"]).sum(round(3)))


# check geocodes
data = pd.read_csv("powerplants_aggregated.csv")
weather = pd.read_csv("../data/geodata/temperature.csv")
buslist = data.bus.unique()
n = 0
for bus in data.bus.unique():
    if bus not in weather.columns:
        n+=1
        print(f"WARNING: bus {bus} not found in weather data")
print(f"{n} of {len(buslist)} busses not found")
