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

# Options for generating data
FORCE = True # force regeneration of all files
STATUS_TO_INCLUDE = ["ONLINE"] # powerplant statuses to include

# Regenerate check
if not os.path.exists("powerplants_aggregated.csv") or FORCE:

    # Load WECC model
    data = json.load(open("wecc240.json","r"))
    assert(data["application"]=="gridlabd")
    objects = data["objects"]

    # Generate individual plant data
    plants = []
    nodelist = {}
    with open("powerplants_split.csv","w") as fh:
        n = 0
        for name,properties in [(x,y) for x,y in objects.items() if y["class"] == "powerplant"]:
            n += 1
            try:
                if properties["status"] not in STATUS_TO_INCLUDE:
                    continue
                node = properties["parent"]
                if "gencap" not in objects[node]:
                    objects[node]["gencap"] = 0.0
                gen = properties["generator"]
                # print(name,"->",node,gen,file=sys.stderr)
                cap = float(properties["operating_capacity"].split()[0])
                objects[node]["gencap"] += cap
                lat = float(objects[node]["latitude"])
                lon = float(objects[node]["longitude"])
                geo = utils.geohash(lat,lon)
                nodelist[geo] = node
                for item in gen.split("|"):
                    plants.append([name,node,geo,item,cap,0,1])
                    # print(name,geo,item,cap,cap/pmax if pmax>0 else float('nan'),1,sep=",",file=fh)
            except:
                e_type,e_value,e_trace = sys.exc_info()
                print(f"ERROR [{name}]: {e_type.__name__} {e_value}",file=sys.stderr)
        print(f"{n} powerplants found")

        # compute total capacities
        totals = {}
        for plant in plants:
            bus,cap,units = plant[2],plant[4],plant[6]
            if not plant[2] in totals:
                totals[bus] = 0
            totals[bus] += cap
        for plant in plants:
            bus = plant[2]
            plant[5] = round(float(plant[4])/float(totals[bus]),3)

        # save results
        data = pd.DataFrame(data=plants,columns=["name","node","bus","gen","cap","cf","units"])
        data.to_csv("powerplants_split.csv",index=False,header=True)

    # load and summarize plant data
    data = pd.read_csv("powerplants_split.csv",index_col=["bus","gen"],usecols=["bus","gen","cap","cf","units"])
    data.groupby(["bus","gen"]).sum().round(3).to_csv("powerplants_aggregated.csv")

# check geocodes
data = pd.read_csv("powerplants_aggregated.csv",index_col=["bus"])
weather = pd.read_csv("../data/geodata/temperature.csv")
buslist = data[data.gen.isin(["WT","PV"])].index.unique()
n = 0
for bus in buslist:
    for gen in data.loc[bus].gen:
        if gen in ["WT","PV"] and bus not in weather.columns:
            n+=1
            print(f"WARNING: bus {bus} not found in weather data for gen type {'|'.join([x for x in data.loc[bus].gen.tolist() if x in ['PV','WT']])}")
            break
print(f"{n} of {len(buslist)} busses not found in weather data")

# generate aggregate powerplant model
n = 0
with open("powerplant_aggregated.glm","w") as fh:
    for bus,plant in data.iterrows():
        n += 1
        gen = nodelist[bus].replace("_N_","_G_")
        geo = utils.geocode(bus)
        print(f"""object powerplant
{{
    name "{gen}_{plant.gen}";
    parent "{gen}";
    latitude {geo[0]};
    lontitude {geo[1]};
    generator {plant.gen};
    operating_capacity {plant.cap} MW;
}}
""",file=fh)
print(f"{n} powerplant objects saved")
