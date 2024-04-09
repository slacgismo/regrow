"""Commercial building loadshapes

This script updates the files in the com_loadshapes folder.
"""

import os,sys
import pandas as pd
import config
import states
import urllib

pd.options.display.max_columns = None
pd.options.display.width = None

building_types = [
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

comstock_server = "https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2021/comstock_amy2018_release_1/timeseries_aggregates/by_county"
comstock_data = "{repo}/state={usps}/g{fips}{puma}-{type}.csv"

os.makedirs("com_loadshapes",exist_ok=True)

with open("com_loadshapes/floorareas.csv","w") as fh:
    print("county,building_type,floorarea[sf]",file=fh)
    for state_usps,state_fips in [(y,z) for x,y,z in states.state_codes if x in config.state_list]:
        for n in range(10,99999,20):
            file = f"com_loadshapes/G{state_fips}{n:05d}.csv.zip"
            result = []
            for type in building_types:
                try:
                    url = comstock_data.format(repo=comstock_server,usps=state_usps,fips=state_fips,puma=f"{n:05d}",type=type)
                    data = pd.read_csv(url,
                        index_col=["timestamp","in.building_type"],
                        usecols = ["timestamp","in.building_type","floor_area_represented",
                            "out.electricity.cooling.energy_consumption","out.electricity.heating.energy_consumption",
                            "out.electricity.exterior_lighting.energy_consumption","out.electricity.fans.energy_consumption",
                            "out.electricity.heat_recovery.energy_consumption","out.electricity.heat_rejection.energy_consumption",
                            "out.electricity.interior_equipment.energy_consumption","out.electricity.interior_lighting.energy_consumption",
                            "out.electricity.pumps.energy_consumption","out.electricity.refrigeration.energy_consumption",
                            "out.electricity.water_systems.energy_consumption","out.site_energy.total.energy_consumption",
                            ],
                        low_memory=True)
                    print("Processing",os.path.basename(url),end="...",flush=True,file=sys.stderr)
                    for column in data.columns:
                        if column.startswith("out."):
                            data[column] = (data[column]/data.floor_area_represented*1000/0.25).round(3)
                    print(f"G{state_fips}{n:05d},{type},{round(data.floor_area_represented.mean(),0)}",file=fh,flush=True)
                    data.drop("floor_area_represented",axis=1,inplace=True)
                    data.index.names = ["datetime","building_type"]
                    data.columns = ["cooling[W/sf]","heating[W/sf]","extlight[W/sf]",
                        "fans[W/sf]","heat_recovery[W/sf]","heat_rejection[W/sf]",
                        "equipment[W/sf]","lighting[W/sf]","pumps[W/sf]","refrigeration[W/sf]",
                        "watersystems[W/sf]","totalenergy[W/sf]"]
                    result.append(data)
                    print("done",file=sys.stderr)
                except urllib.error.HTTPError:
                    pass
            if result and not os.path.exists(file):
                print("Saving",f"G{state_fips}{n:05d}",end="...",flush=True,file=sys.stderr)
                pd.concat(result).to_csv(file,index=True,header=True,compression="zip")
                print("done",file=sys.stderr)
            else:
                break
