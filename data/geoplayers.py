import os, sys
import pandas as pd
from nsrdb_weather import geohash
import config

pd.options.display.max_columns = None
pd.options.display.width = 1024

REFRESH = False

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
for file in sorted(os.listdir("com_loadshapes")):
    # result = {}
    # for variable in variables:
    #     result[variable] = pd.DataFrame(pd.date_range("2018-01-01 00:00:00","2023-01-01 00:00:00",freq="15min"))
    result = None
    if file.startswith("G") and file.endswith(".csv.zip"):
        usps = file[1:3]
        fips = file[3:8]
        geocode = counties.loc[usps+fips[1:4]].geocode
        csv = os.path.join("geodata",geocode,"commercial.csv.zip")
        if os.path.exists(csv) or REFRESH:
            continue
        print("Processing",usps,fips,geocode,end="...",flush=True,file=sys.stderr)
        os.makedirs(os.path.join("geodata",geocode),exist_ok=True)
        data = pd.DataFrame(pd.read_csv(os.path.join("com_loadshapes",file),
            index_col=["datetime","building_type"],
            ).groupby("datetime").sum().sum(axis=1).round(1))
        # TODO: weight by SF
        data.columns = ["power[kW]"]
        try:
            data.to_csv(csv,header=True,index=True,compression="zip")
        except:
            os.remove(csv)
        # for variable in variables:
        #     result[variable][geocode] = data[variable]
        # for variable in variables:
        print("done",file=sys.stderr)



