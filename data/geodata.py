"""Building load geodata

Syntax: python3 -m geodata OPTION [...]

Options
-------

    --freq=FREQ     set the data resample frequency (default is 1H)
    -h|--help|help  generate this help document
    --inputs        generate list of input files
    --outputs       generate list of output files
    --refresh       refresh intermediate files before updating
    --update        update only missing/outdates files
    --verbose       verbose progress updates

Description
-----------

This script generates the building load and weather geodata for the nodes in the system.
"""

import os,sys
import pandas as pd
import config
import states
import urllib

INPUTS = {
    "NODES" : "wecc240_gis.csv",
    "COUNTIES" : "counties.csv",
}

INTERNAL = {
    "COOLING" : "geodata/counties/cooling.csv",
    "HEATING" : "geodata/counties/heating.csv",
    "SOLAR" : "geodata/counties/solar.csv",
    "TEMPERATURE" : "geodata/counties/temperature.csv",
    "TOTAL" : "geodata/counties/total.csv",
    "WIND" : "geodata/counties/wind.csv",
}

OUTPUTS = {
    "COOLING" : "geodata/cooling.csv",
    "HEATING" : "geodata/heating.csv",
    "SOLAR" : "geodata/solar.csv",
    "TEMPERATURE" : "geodata/temperature.csv",
    "TOTAL" : "geodata/total.csv",
    "WIND" : "geodata/wind.csv",
}

REFRESH = False
FREQ = "1h"

from utils import *

options.context = "geodata.py"
pd.options.display.max_columns = None
pd.options.display.width = None

for arg in read_args(sys.argv,__doc__):
    if arg.startswith("--freq"):
        FREQ = arg.split("=")[1]
    elif arg == "--refresh":
        REFRESH = True
    elif arg != "--update":
        raise Exception(f"option '{arg}' is not valid")

commercial_buildings = [
    "fullservicerestaurant",
    "hospital",
    "largehotel",
    "largeoffice",
    "mediumoffice",
    "outpatient",
    "primaryschool",
    "quickservicerestaurant",
    "retailstandalone",
    "retailstripmall",
    "secondaryschool",
    "smallhotel",
    "smalloffice",
    "warehouse",
]

residential_buildings = [
    "mobile_home",
    "multi-family_with_2_-_4_units",
    "multi-family_with_5plus_units",
    "single-family_attached",
    "single-family_detached",
]

comstock_server = "https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2021/comstock_amy2018_release_1/timeseries_aggregates/by_county"
comstock_data = "{repo}/state={usps}/g{fips}0{puma}0-{type}.csv"

resstock_server = "https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2021/resstock_amy2018_release_1/timeseries_aggregates/by_county"
resstock_data = "{repo}/state={usps}/g{fips}0{puma}0-{type}.csv"

weather_server = "https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2021/resstock_amy2018_release_1/weather/amy2018"
weather_data = "{repo}/G{fips}0{puma}0_2018.csv"

if __name__ == "__main__":

    ok = True
    for file in OUTPUTS.values():
        ok |= os.path.exists(file)
    if ok and not REFRESH:
        exit(0)
        
    counties = pd.read_csv(INPUTS["COUNTIES"],
        converters = {"fips":str},
        index_col = ["fips"],
        )
    
    geodata = dict(
        temperature = [],
        wind = [],
        solar = [],
        cooling = [],
        heating = [],
        total = [],
        )
    sources = dict(
        temperature = "temperature[degC]",
        wind = "wind[m/s]",
        solar = "solar[W/m^2]",
        cooling = "cooling[MW]",
        heating = "heating[MW]",
        total = "total[MW]",
        )

    verbose("Loading existing county geodata",end="...")
    for table in geodata:
        try:
            data = pd.read_csv(INTERNAL[table.upper()],index_col=["timestamp"],parse_dates=["timestamp"])
            for column in data:
                geodata[table].append(pd.DataFrame(data=data[column].values,index=data.index,columns=[column]))
        except:
            geodata[table] = []
    verbose("ok")

    for state_usps,state_fips in [(states.state_codes_byname[x]["usps"],states.state_codes_byname[x]["fips"]) for x in config.state_list if x in states.state_codes_byname]:
        for puma in [x for x in counties.index.values if x.startswith(state_fips)]:

            geocode = counties.loc[puma]['geocode']

            # check if data exists already
            found = 0
            for table in geodata:
                if geocode in [list(x.columns)[0] for x in geodata[table]]:
                    found += 1
            if found == len(geodata):
                continue # data is up-to-date

            verbose(f"Processing {counties.loc[puma]['county']} {counties.loc[puma]['usps']}",end="...")

            # weather data
            url = weather_data.format(repo=weather_server,fips=state_fips,puma=puma[2:])
            data = pd.read_csv(url,
                index_col = [0],
                usecols = [0,1,3,5],
                parse_dates = [0],
                dtype = float,
                low_memory = True,
                header=None,
                skiprows=1,
                )
            data.columns = ["temperature[degC]","wind[m/s]","solar[W/m^2]"]
            data.index.name = "timestamp"
            weather = data.resample(FREQ).mean()

            # residential buildings
            buildings = []
            for building_type in residential_buildings:
                url = resstock_data.format(repo=resstock_server,usps=state_usps,fips=state_fips,puma=puma[2:],type=building_type)
                try:

                    data = pd.read_csv(url,
                        index_col=["timestamp"],
                        usecols = ["timestamp",
                            "in.geometry_building_type_recs",
                            "out.electricity.cooling.energy_consumption",
                            "out.electricity.heating.energy_consumption",
                            "out.electricity.heating_supplement.energy_consumption",
                            "out.electricity.total.energy_consumption",
                            ],
                        parse_dates = ["timestamp"],
                        converters = {
                            "out.electricity.cooling.energy_consumption" : lambda x: float(x)/1000,
                            "out.electricity.heating.energy_consumption" : lambda x: float(x)/1000,
                            "out.electricity.heating_supplement.energy_consumption" : lambda x: float(x)/1000,
                            "out.electricity.total.energy_consumption" : lambda x: float(x)/1000,
                        },
                        low_memory=True)
                    verbose(".",end="")
                    data.columns = ["building_type","cooling[MW]","heating[MW]","auxheat[MW]","total[MW]"]
                    data["heating[MW]"] += data["auxheat[MW]"]
                    data.drop("auxheat[MW]",axis=1,inplace=True)
                    data = pd.DataFrame(data.resample(FREQ).sum())
                    buildings.append(data.reset_index())

                except urllib.error.HTTPError as err:
                    
                    pass

            # commercial buildings
            for building_type in commercial_buildings:
                url = comstock_data.format(repo=comstock_server,usps=state_usps,fips=state_fips,puma=puma[2:],type=building_type)
                try:

                    data = pd.read_csv(url,
                        index_col=["timestamp"],
                        usecols = ["timestamp",
                            "in.building_type",
                            "out.electricity.cooling.energy_consumption",
                            "out.electricity.heating.energy_consumption",
                            "out.electricity.total.energy_consumption",
                            ],
                        parse_dates = ["timestamp"],
                        converters = {
                            "out.electricity.cooling.energy_consumption" : lambda x: float(x)/1000,
                            "out.electricity.heating.energy_consumption" : lambda x: float(x)/1000,
                            "out.electricity.total.energy_consumption" : lambda x: float(x)/1000,
                        },
                        low_memory=True)
                    verbose(".",end="")
                    data.columns = ["building_type","cooling[MW]","heating[MW]","total[MW]"]
                    data = pd.DataFrame(data.resample(FREQ).sum())
                    buildings.append(data.reset_index())

                except urllib.error.HTTPError as err:
                    
                    pass

            try:

                buildings = pd.DataFrame(pd.concat(buildings).set_index(["timestamp","building_type"]).groupby("timestamp").sum())

                for group,prec in [[weather,1],[buildings,3]]:
                    for table,column in sources.items():
                        if column in group.columns:
                            geodata[table].append(pd.DataFrame(data=group[column].values,index=group.index,columns=[geocode]).round(prec))

                verbose("ok")

            except Exception as err:

                verbose(err)

            # save progress
            os.makedirs("geodata/counties",exist_ok=True)
            for table,data in geodata.items():
                pd.concat(data,axis=1).to_csv(INTERNAL[table.upper()],index=True,header=True)

    # final data consolidation for node processing
    verbose("Consolidating county geodata tables",end="...")
    for table,data in list(geodata.items()):
        geodata[table] = pd.concat(data,axis=1)
    verbose("ok")

    # load the node list
    verbose("Generating node geodata",end="...")

    node_geodata = dict([(x,[]) for x in sources])
    nodes = pd.read_csv(INPUTS["NODES"],index_col=[0],usecols=["Bus  Number","Lat","Long"])
    node_geocodes = [geohash(x,y,6) for x,y in nodes[["Lat","Long"]].values]
    node_map = dict([(x,[]) for x in node_geocodes])
    for county in geodata[list(geodata)[0]].columns:
        node_map[nearest(county,node_geocodes)].append(county)
    # print(node_map)
    for node,county_list in node_map.items():
        if not county_list:
            continue
        closest_county = nearest(node,county_list)
        # print(node,"-->",closest_county,county_list)
        for table in node_geodata:
            if not sources[table].endswith("[MW]"): # nearest table
                if len(node_geodata[table]) == 0:
                    node_geodata[table] = pd.DataFrame(data=geodata[table][closest_county].values,index=geodata[table][closest_county].index,columns=[node])
                else:
                    node_geodata[table][node] = pd.DataFrame(data=geodata[table][closest_county].values,index=geodata[table][closest_county].index,columns=[node])
                continue
            else:
                for county in county_list:
                    if county not in geodata[table].columns:
                        warning(f"county {county} not found in geodata {table}")
                        continue
                    if len(node_geodata[table]) == 0:
                        # print("Creating",node,table,"using",county,"\n",geodata[table][county],)
                        node_geodata[table] = pd.DataFrame(data=geodata[table][county].values,index=geodata[table][county].index,columns=[node])
                    elif not node in node_geodata[table].columns:
                        # print("Appending",county,table,"to",node,"\n",geodata[table][county])
                        node_geodata[table][node] = geodata[table][county]
                    else:
                        # print("Adding",county,table,"to",node,"\n",geodata[table][county])
                        node_geodata[table][node] += geodata[table][county]
                    # print("  -->",node,"\n",node_geodata[table])

    for table,data in node_geodata.items():
        data.round(2).to_csv(f"geodata/{table}.csv",index=True,header=True)
    verbose("ok")
