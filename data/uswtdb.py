"""Download DATA data for WECC"""

import os
import sys
import json
sys.path.append("../data")
import utils
import pandas as pd
import gridlabd.census as census
import requests
import zipfile
import io

pd.options.display.width = None
pd.options.display.max_columns = None

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

BUSLIST = pd.read_csv("../data/geodata/solar.csv",index_col=[0]).columns.tolist()

# Assemble county data
COUNTIES = []
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
    df.columns = [x.lower() for x in df.columns]
    if state in INCLUDE:
        df.drop(df.loc[~df['countyfp'].isin(INCLUDE[state])].index,inplace=True,axis=0)
    rename = {"couname":"name"}
    df.columns = [rename[x] if x in rename else x for x in df.columns]
    df["fips"] = df["statefp"] + df["countyfp"]
    df["state"] = state
    df.drop(["statefp","countyfp"],axis=1,inplace=True)
    df.set_index("fips",inplace=True)
    # df["county"] = df.name + " " + df.state
    df.rename({"name":"county"},axis=1,inplace=True)
    df.drop(["stname","population"],axis=1,inplace=True)

    # Save for concat
    COUNTIES.append(df)

COUNTIES = pd.concat(COUNTIES).drop(EXCLUDE,axis=0).sort_index()
COUNTIES["bus"] = [utils.nearest(utils.geohash(x,y),BUSLIST) for x,y in COUNTIES[["latitude","longitude"]].values]
COUNTIES.drop(["latitude","longitude"],inplace=True,axis=1)

zipdata = zipfile.ZipFile(io.BytesIO(requests.get("https://energy.usgs.gov/uswtdb/assets/data/uswtdbCSV.zip").content))
DATA = pd.read_csv(zipdata.open([x for x in zipdata.namelist() if x.endswith(".csv")][0],"r"),
    usecols = ["t_fips","ylat","xlong","p_name","p_year","t_model","t_cap","t_hh","t_rd"],
    )
DATA.dropna(subset=["t_cap"],inplace=True)
DATA["t_cap"] /= 1000
DATA.columns =  ["fips","name","year","gentype","capacity[MW]","hub_height[m]","rotor_diameter[m]","longitude","latitude"]
DATA["fips"] = [f"{x:05.0f}" for x in DATA["fips"]]
DATA.set_index(["fips"],inplace=True)
DATA.sort_index(inplace=True)

DATA = DATA.join(COUNTIES,how="inner").reset_index().set_index("bus").sort_index()
DATA["county"] = [f"{x} {y}" for x,y in DATA[["county","state"]].values]
DATA.drop("state",inplace=True,axis=1)
DATA.to_csv("uswtdb.csv",index=True,header=True)
