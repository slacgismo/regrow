"""Commercial building loadshapes

This script updates the files in the com_loadshapes folder.
"""

import os,sys
import pandas as pd
import config
import states
import urllib
from utils import *
options.context = "commercial.py"
options.verbose = True

FREQ = "1h"

pd.options.display.max_columns = None
pd.options.display.width = None

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

    counties = pd.read_csv("counties.csv",
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
    for state_usps,state_fips in [(states.state_codes_byname[x]["usps"],states.state_codes_byname[x]["fips"]) for x in config.state_list if x in states.state_codes_byname]:
        for puma in [x for x in counties.index.values if x.startswith(state_fips)]:
            geocode = counties.loc[puma]['geocode']

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
                    for item,column in sources.items():
                        if column in group.columns:
                            geodata[item].append(pd.DataFrame(data=group[column].values,index=group.index,columns=[geocode]).round(prec))

                verbose("ok")

            except Exception as err:

                verbose(err)

            verbose(f"Saving geodata",end="...")
            os.makedirs("geodata",exist_ok=True)

            for item,data in geodata.items():
                pd.concat(data,axis=1).to_csv(f"geodata/{item}.csv",index=True,header=True)
                verbose(".",end="")

            verbose("ok")



    # for state_usps,state_fips in [(y,z) for x,y,z in states.state_codes if x in config.state_list]:
    #     print(state_usps,state_fips)

    # with open("commercial/floorareas.csv","w") as fh:
    #     print("county,building_type,floorarea[sf]",file=fh)
    #     for state_usps,state_fips in [(y,z) for x,y,z in states.state_codes if x in config.state_list]:
    #         for n in range(10,99999,20):
    #             file = f"commercial/G{state_fips}{n:05d}.csv.zip"
    #             result = []
    #             for type in commercial_buildings:
    #                 try:
    #                     url = comstock_data.format(repo=comstock_server,usps=state_usps,fips=state_fips,puma=f"{n:05d}",type=type)
    #                     data = pd.read_csv(url,
    #                         index_col=["timestamp","in.building_type"],
    #                         usecols = ["timestamp","in.building_type","floor_area_represented",
    #                             "out.electricity.cooling.energy_consumption","out.electricity.heating.energy_consumption",
    #                             "out.electricity.exterior_lighting.energy_consumption","out.electricity.fans.energy_consumption",
    #                             "out.electricity.heat_recovery.energy_consumption","out.electricity.heat_rejection.energy_consumption",
    #                             "out.electricity.interior_equipment.energy_consumption","out.electricity.interior_lighting.energy_consumption",
    #                             "out.electricity.pumps.energy_consumption","out.electricity.refrigeration.energy_consumption",
    #                             "out.electricity.water_systems.energy_consumption","out.site_energy.total.energy_consumption",
    #                             ],
    #                         low_memory=True)
    #                     print("Processing",os.path.basename(url),end="...",flush=True,file=sys.stderr)
    #                     for column in data.columns:
    #                         if column.startswith("out."):
    #                             data[column] = (data[column]/data.floor_area_represented*1000/0.25).round(3)
    #                     print(f"G{state_fips}{n:05d},{type},{round(data.floor_area_represented.mean(),0)}",file=fh,flush=True)
    #                     data.drop("floor_area_represented",axis=1,inplace=True)
    #                     data.index.names = ["datetime","building_type"]
    #                     data.columns = ["cooling[W/sf]","heating[W/sf]","extlight[W/sf]",
    #                         "fans[W/sf]","heat_recovery[W/sf]","heat_rejection[W/sf]",
    #                         "equipment[W/sf]","lighting[W/sf]","pumps[W/sf]","refrigeration[W/sf]",
    #                         "watersystems[W/sf]","totalenergy[W/sf]"]
    #                     result.append(data)
    #                     print("done",file=sys.stderr)
    #                 except urllib.error.HTTPError:
    #                     pass
    #             if result and not os.path.exists(file):
    #                 print("Saving",f"G{state_fips}{n:05d}",end="...",flush=True,file=sys.stderr)
    #                 pd.concat(result).to_csv(file,index=True,header=True,compression="zip")
    #                 print("done",file=sys.stderr)
    #             else:
    #                 break

    # os.makedirs("commercial",exist_ok=True)

