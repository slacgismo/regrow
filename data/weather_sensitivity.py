import os, sys
import pandas as pd
pd.options.display.max_columns = None
pd.options.display.width = 1024

Theat = 60
Tcool = 80
server = "https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2021/resstock_amy2018_release_1/weather/amy2018/"
pumas = {
    "Alameda" : "G0600010",
    "Contra Costa" : "G0600130",
    "Los Angeles" : "G0600370",
    "Riverside" : "G0600650",
    "San Bernadino" : "G0600710",
    "San Diego" : "G0600730",
    "San Francisco" : "G0600750",
    "San Mateo" : "G0600810",
    "Santa Clara" : "G0600850",
}

print("Loading loadshapes...",flush=True)
loadshapes = pd.read_csv("loadshapes.csv.zip",
    index_col = ["county","building_type","datetime"],
    )

for county,puma in pumas.items():
    file = f"weather/CA-{county.replace(' ','_')}.csv.zip"
    if not os.path.exists(file):
        print("Downloading",file,"...",flush=True)
        data = pd.read_csv(os.path.join(server,f"{puma}_2018.csv"),
            index_col = ["date_time"],
            )
        data.index.name = "datetime"
        data.columns = ["TD[degC]","RH[%]","WS[m/s]","WD[deg]","GH[W/m^2]","DN[W/m^2]","DH[W/m^2]"]
        data.round(1).to_csv(file,index=True,header=True,compression="zip")
    else:
        print("Loading weather for",file,"...",flush=True)
    data = pd.read_csv(file,index_col="datetime")

    heating = (data["TD[degC]"]<Theat).index
    cooling = (data["TD[degC]"]>Tcool).index
    solar = (data["DN[W/m^2]"]>0).index

