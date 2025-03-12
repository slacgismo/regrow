"""Split and reaggregate powerplants by WECC node

This script reads the WECC240 model and splits up powerplant objects based on
the generator type. The operating capacities are summed and their
contribution to the total generating capacity is totaled.

The input file is `wecc240_raw.json` after it is compiled by GridLAB-D from
the original data. The intermediate file is `powerplants_split.csv`. The
output files are `powerplants_aggregated.csv` and
`powerplants_aggregated.glm`.

Output CSV fields are as follows:

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

pd.options.display.width = None
pd.options.display.max_columns = None

# Options for generating data
STATUS_TO_INCLUDE = ["ONLINE"] # powerplant statuses to include

# Load WECC240 raw PSSE model
if os.system("gridlabd -D suppress_repeat_messages=FALSE -C wecc240_psse.glm ../data/wecc240_gis.glm ../data/powerplants.glm ../data/powerplants_gis.glm -o wecc240_raw.json 1>wecc240_raw.txt 2>&1") != 0:
    print("ERROR: gridlabd model compile failed",file=sys.stderr)
    quit(1)
data = json.load(open("wecc240_raw.json","r"))
assert(data["application"]=="gridlabd")
objects = data["objects"]

GENCOSTS = json.load(open("gencosts.json","r"))

# Generate individual plant data
unknowns = pd.read_csv("powerplants_unknown.csv",index_col=["name"]).sort_index()

plants = []
nodelist = {}
with open("powerplants_split.csv","w") as fh:
    n = 0

    # gather non-renewable facilities
    for name,properties in [(x,y) for x,y in objects.items() if y["class"] == "powerplant"]:
        n += 1
        try:
            if properties["status"] not in STATUS_TO_INCLUDE:
                continue
            node = properties["parent"]
            if "gencap" not in objects[node]:
                objects[node]["gencap"] = 0.0
            gen = properties["generator"]
            cap = float(properties["operating_capacity"].split()[0])
            objects[node]["gencap"] += cap
            lat = float(objects[node]["latitude"])
            lon = float(objects[node]["longitude"])
            geo = utils.geohash(lat,lon)
            nodelist[geo] = node
            types = [x for x in gen.split("|") if x not in ["PV"]] # ignore PV and WT facilities (add later)
            for gentype in types:
                if gentype == "UNKNOWN":
                    gentype = unknowns.loc[name].gentype
                plants.append([name,node,geo,gentype,round(cap/len(types),1),0,1])
        except:
            e_type,e_value,e_trace = sys.exc_info()
            print(f"ERROR [{name}]: {e_type.__name__} {e_value}",file=sys.stderr)
    print("INFO:",f"{n} powerplants found with status {'|'.join(STATUS_TO_INCLUDE)}")

    # add PV facilities from USPVDB
    pvgens = pd.read_csv("../data/uspvdb.csv").set_index("name")
    print("INFO:",len(pvgens),"photovoltaic facilities added")
    for name,row in pvgens.iterrows():
        geo = utils.nearest(row["bus"],nodelist)
        plants.append([name,nodelist[geo],geo,"PV",row["capacity[MW]"],0,1])

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
data = pd.read_csv("powerplants_split.csv",index_col=["name"],usecols=["bus","gen","name","cap","cf","units"]).sort_index()
data.to_csv("powerplants_data.csv",index=True,header=True)
names = data.reset_index().set_index(["bus","gen","name"]).sort_index().groupby(level=[0,1,2]).sum()
data = pd.read_csv("powerplants_split.csv",index_col=["bus","gen"],usecols=["bus","gen","cap","cf","units"])
data.groupby(["bus","gen"]).sum().round(3).to_csv("powerplants_aggregated.csv")

# check geocodes
data = pd.read_csv("powerplants_aggregated.csv",index_col=["bus"])
weather = pd.read_csv("../data/geodata/temperature.csv")
buslist = data[data.gen.isin(["WT","PV"])].index.unique()
n = 0
gentypes = {}
gensizes = {}
for bus in buslist:
    gens = data.loc[bus].gen if not isinstance(data.loc[bus].gen,str) else [data.loc[bus].gen]
    for gen in gens:
        if gen not in gentypes:
            gentypes[gen] = []
            gensizes[gen] = 0
        gentypes[gen].append(bus)
        gensizes[gen] += data.loc[bus].cap.sum()
        if gen in ["WT","PV"] and bus not in weather.columns:
            n+=1
            print(f"WARNING: bus {bus} not found in weather data for gen type {'|'.join([x for x in gens if x in ['PV','WT']])}",file=sys.stderr)
            break
print(f"WARNING: {n} of {len(buslist)} powerplant busses not found in weather data",file=sys.stderr)
for gt,bus in gentypes.items():
    print("INFO:",gt,"located at",len(bus),"busses,",round(gensizes[gt]/1000,2),"GW installed capacity")

# generate aggregate powerplant GLM model
n,m = 0,0
with open("powerplants_aggregated.glm","w") as fh:
    print("module pypower;",file=fh)
    for bus,plant in data.iterrows():
        n += 1

        # add gen object if missing
        gen = nodelist[bus].replace("_N_","_G_")
        if gen not in objects:
            m += 1
            objects[gen] = {
                "parent":gen.replace("_G_","_N_"),
                "Pmax": f"{round(totals[bus],1)} MW",
                "Qmax": f"{round(totals[bus]*0.2,1)} MW",
                "Qmin": f"{round(-totals[bus]*0.2,1)} MW",
                }
            properties = "\n".join([f"    {x} {y};" for x,y in objects[gen].items()])
            print(f"""object pypower.gen
{{
    name "{gen}";
{properties}
    status IN_SERVICE;
    object gencost 
    {{
        model POLYNOMIAL;
        startup 0.0 $;
        shutdown 0.0 $;
        costs "0.0,0.0,0.0";
    }};
}}""",file=fh)

        geo = utils.geocode(bus)
        if plant.gen != "UNKNOWN":
            properties = "\n".join([f"    {x} {y};" for x,y in GENCOSTS[plant.gen].items()])
        else:
            print(f"WARNING: {gen} has no cost data, gen type is UNKNOWN",file=sys.stderr)
            properties = "    // no cost data for unknown plant type"
        unitinfo = "\n    // ".join([f"{x}: {', '.join([a+'='+str(b) for a,b in y.items()])}" for x,y in names.loc[(bus,plant.gen)].to_dict('index').items()])
        print(f"""object pypower.powerplant
{{
    // {unitinfo}
    name "{gen}_{plant.gen}";
    parent "{gen}";
    latitude {geo[0]};
    longitude {geo[1]};
    generator {plant.gen};
    operating_capacity {plant.cap} MW;
{properties}
}}
""",file=fh)

print("INFO:",f"{n} powerplant objects generated to GLM")
print("INFO:",f"{m} gen object added to GLM")
