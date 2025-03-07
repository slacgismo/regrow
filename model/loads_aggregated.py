"""Constructed aggregated load models by WECC node

This script reads the WECC240 model and generates loads for each node
based on the available load data from NREL and census data from the US Census
Bureau.

The input file is `wecc240_raw.json` after the model is compiled by GridLAB-D. 
The output files are `loads_aggregated.csv` and `loads_aggregated.glm`.
"""

import os
import sys
import json
sys.path.append("../data")
import utils
import pandas as pd
import gridlabd.census as census

# Counties in WECC
STATES = ["CA","WA","OR","ID","MT","WY","NV","UT","AZ","NM","CO"] 
EXCLUDE = [ # Counties to leave out
    # MT
    '30019', # Daniels
    '30021', # Dawson
    '30025', # Fallon
    '30083', # Richland
    '30085', # Roosevelt
    '30091', # Sheridan
    '30109', # Wibaux
    # NM
    '35009', # Curry
    '35025', # Lea
    '35037', # Quay
    '35041', # Roosevelt
    ] 
INCLUDE = {
    "TX": [
        '141', # El Paso
    ],
    "SD": [
        '033', # Custer
        '047', # Fall River
        '081', # Lawrence
    ]} # counties to add in from other states

# Assemble county data
COUNTY = {}
data = []
for state in STATES + list(INCLUDE):
    fips = f"{census.FIPS_STATES[state]['fips']:02.0f}"

    # County population centroid data
    URL = f"https://www2.census.gov/geo/docs/reference/cenpop2020/county/CenPop2020_Mean_CO{fips}.txt"
    df = pd.read_csv(URL,
        converters = {
            "COUNAME": census.strict_ascii,
            "STATEFP": lambda x: f"{float(x):02.0f}",
            "COUNTYFP": lambda x: f"{float(x):03.0f}",
        },
        usecols = ["STATEFP","COUNTYFP","STNAME","COUNAME","POPULATION","LATITUDE","LONGITUDE"],
        )
    statename = df.STNAME.unique()[0].replace(' ','')

    # County residential units data
    URL = f"https://www2.census.gov/geo/docs/reference/2020addresscountlist/{fips}_{statename}_AddressBlockCountList_012020.txt"
    addr = pd.read_csv(URL,
        usecols=['COUNTY','TOTAL RESIDENTIAL'],
        sep='|',
        converters={
            "COUNTY": lambda x: f"{float(x):03.0f}" if x else "000",
            "TOTAL RESIDENTIAL": lambda x: int(x),
        }
        ).groupby('COUNTY').sum().drop("000",axis=0)
    addr.columns = ["units"]

    # Join population and residences data
    df = df.set_index("COUNTYFP").join(addr).reset_index()

    # Finalize dataframe
    df.columns = [x.lower() for x in df.columns]
    if state in INCLUDE:
        df.drop(df.loc[~df['countyfp'].isin(INCLUDE[state])].index,inplace=True,axis=0)
    rename = {"couname":"name"}
    df.columns = [rename[x] if x in rename else x for x in df.columns]
    df["fips"] = df["statefp"] + df["countyfp"]
    df["state"] = state
    df.drop(["statefp","countyfp"],axis=1,inplace=True)
    df.set_index("fips",inplace=True)
    df['units'] = df['units'].astype(int)
    df['occupancy'] = (df['population'] / df['units']).round(2)

    # Save for concat
    data.append(df)

# Concatenate county datasets
COUNTY = pd.concat(data).drop(EXCLUDE,axis=0).to_dict('index')
# print(len(COUNTY),"counties' population centroid data loaded ok",flush=True)

# Load WECC240 raw PSSE model
if os.system("gridlabd -C wecc240_psse.glm ../data/wecc240_gis.glm ../data/powerplants.glm ../data/powerplants_gis.glm -o wecc240_raw.json") != 0:
    print("ERROR: gridlabd model compile failed",file=sys.stderr)
    quit(1)
data = json.load(open("wecc240_raw.json","r"))
assert(data["application"]=="gridlabd")
objects = data["objects"]

# Load geodata for total load
total_file = "../data/geodata/total.csv"
total = pd.read_csv(total_file,index_col=[0],parse_dates=True)

# Map counties to geodata total load
geocodes = {}
for name,data in [(x,y) for x,y in objects.items() if y['class'] == 'load']:
    node = objects[data["parent"]]
    objects[name]['geocode'] = utils.geohash(float(node['latitude']),float(node['longitude']))
    geonode = utils.nearest(objects[name]['geocode'],total.columns)
    geocodes[geonode] = {"name":name,"population":0.0,"units":0.0}
    objects[name]['geonode'] = geonode

# Generate update
for county in COUNTY.values():
    county['geocode'] = utils.geohash(county['latitude'],county['longitude'])
    county['geonode'] = utils.nearest(county['geocode'],geocodes)
    # print(county['state'],county['name'],county['units'],county['residents_per_unit'],county['geocode'],county['geonode'],round(utils.distance(county['geocode'],county['geonode'],'mile'),1),"miles")
    geocodes[county['geonode']]["population"] += county["population"]
    geocodes[county['geonode']]["units"] += county["units"]

# print(len(COUNTY),"counties loaded")
with open("loads_aggregated.glm","w") as glm:
    for geocode,data in sorted(geocodes.items()):
        if data["units"] > 0:
            print(f"""object building
{{
    parent "{data["name"]}";
    type RESIDENTIAL;
    units {int(data["units"])};
    occupancy {data["population"]/data["units"]:.2f};
}}""",file=glm)
