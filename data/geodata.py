"""Update geodata files"""

import os, sys
import pandas as pd
from nsrdb_weather import geohash
import config

pd.options.display.max_columns = None
pd.options.display.width = 1024

REFRESH = True # force regeneration of all results from original data
COMPRESS = False # compress output data files

os.makedirs("geodata",exist_ok=True)

#
# Update the counties geodata from the Census bureau data
#
if not os.path.exists("counties.csv") or REFRESH:
    counties = pd.read_csv("https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2019_Gazetteer/2019_Gaz_counties_national.zip",
        delimiter = '\t',
        usecols = [0,1,3,8,9],
        skiprows = [0],
        names = ["usps","fips","county","latitude","longitude"],
        )
    counties["fips"] = [f"{x:05d}" for x in counties["fips"]]
    counties["geocode"] = [geohash(float(x[0]),float(x[1])) for x in zip(counties["latitude"],counties["longitude"])]
    counties.to_csv("counties.csv",index=False,header=True)
else:
    counties = pd.read_csv("counties.csv",dtype={"fips":str})
counties.set_index("fips",inplace=True)

#
# Read commercial building loadshapes
#
variables = ["cooling[W/sf]","heating[W/sf]","extlight[W/sf]",
            "fans[W/sf]","heat_recovery[W/sf]","heat_rejection[W/sf]",
            "equipment[W/sf]","lighting[W/sf]","pumps[W/sf]","refrigeration[W/sf]",
            "watersystems[W/sf]","totalenergy[W/sf]"]

floorareas = pd.read_csv("com_loadshapes/floorareas.csv",index_col=["county","building_type"])

for file in sorted(os.listdir("com_loadshapes")):
    result = None
    if file.startswith("G") and file.endswith(".csv.zip"):

        # identify and check target CSV file
        usps = file[1:3]
        fips = file[3:8]
        geocode = counties.loc[usps+fips[1:4]].geocode
        csv = os.path.join("geodata",geocode+(".csv.zip" if COMPRESS else ".csv"))
        if os.path.exists(csv) and not REFRESH:
            continue

        # load commercial building data
        print("Processing",usps,fips,geocode,end="...",flush=True,file=sys.stderr)
        data = pd.read_csv(os.path.join("com_loadshapes",file),
            index_col=["datetime","building_type"],
            converters = {"building_type":str.lower},
            ).join(floorareas)
        columns = []
        drops = []
        for column in data.columns:
            if column.endswith("[W/sf]"):
                data[column] *= data["floorarea[sf]"]/1e9
                columns.append(column.replace("[W/sf]","[MW]"))
            else:
                drops.append(column)
        data.drop(drops,axis=1,inplace=True)
        data.columns = columns
        
        result = pd.DataFrame(data.groupby("datetime").sum().sum(axis=1).round(3))
        result.columns = ["commercial_buildings[MW]"]

        # save 
        try:
            result.to_csv(csv,header=True,index=True,compression=("zip" if COMPRESS else None))
        except:
            os.remove(csv)

        print("done",file=sys.stderr)



