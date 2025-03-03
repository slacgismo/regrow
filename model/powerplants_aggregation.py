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
import sys
import json
sys.path.append("../data")
import utils

data = json.load(open("wecc240.json","r"))
assert(data["application"]=="gridlabd")
objects = data["objects"]
with open("powerplants_split.csv","w") as fh:
    print("name,bus,gen,cap,cf,units",file=fh)
    for name,properties in [(x,y) for x,y in objects.items() if y["class"] == "powerplant"]:
        bus = properties["parent"]
        gen = properties["generator"]
        cap = float(properties["operating_capacity"].split()[0])
        pmax = float(objects[bus]["Pmax"].split()[0])
        node = objects[bus]["parent"]
        lat = float(objects[node]["latitude"])
        lon = float(objects[node]["longitude"])
        geo = utils.geohash(lat,lon)
        for item in gen.split("|"):
            print(name,geo,item,cap,cap/pmax if pmax>0 else float('nan'),1,sep=",",file=fh)

import pandas as pd
data = pd.read_csv("powerplants_split.csv",index_col=["bus","gen"],usecols=["bus","gen","cap","cf","units"])
data.groupby(["bus","gen"]).sum().round(3).to_csv("powerplants_aggregated.csv")