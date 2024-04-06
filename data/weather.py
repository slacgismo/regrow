import marimo

__generated_with = "0.3.9"
app = marimo.App()


@app.cell
def __():
    #
    # Settings and configurations
    #

    Theat = 30 # heating cut-off temperature
    Tcool = 20 # cooling cut-off temperature

    # data source
    server = "https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2021/resstock_amy2018_release_1/weather/amy2018/"

    # county puma codes
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
    return Tcool, Theat, pumas, server


@app.cell
def __(pd):
    #
    # Loadshape data
    #
    loadshapes = pd.read_csv("loadshapes.csv.zip",
        index_col = ["county","building_type","datetime"],
        parse_dates = ["datetime"],
        )
    return loadshapes,


@app.cell
def __(os, pd, pumas, server):
    weather = {}
    for county,puma in pumas.items():
        file = f"weather/CA-{county.replace(' ','_')}.csv.zip"
        if not os.path.exists(file):
            print("Downloading",file,end="...",flush=True)
            _data = pd.read_csv(os.path.join(server,f"{puma}_2018.csv"),
                index_col = ["date_time"],
                )
            _data.index.name = "datetime"
            _data.columns = ["TD[degC]","RH[%]","WS[m/s]","WD[deg]","GH[W/m^2]","DN[W/m^2]","DH[W/m^2]"]
            _data.round(1).to_csv(file,index=True,header=True,compression="zip")
            print("done",flush=True)
        weather[county] = pd.read_csv(file,
                                      index_col="datetime",
                                      parse_dates = ["datetime"],
                                     )
    return county, file, puma, weather


@app.cell
def __(loadshapes):
    loadshapes.columns
    return


@app.cell
def __(Tcool, Theat, county, loadshapes, np, plt, pumas, weather):
    for _county,_puma in pumas.items():
        _data = weather[_county]
        _heating = _data[_data["TD[degC]"]<Theat]["TD[degC]"]
        _cooling = _data[_data["TD[degC]"]>Tcool]["TD[degC]"]
        _solar = _data[_data["DN[W/m^2]"]>0]["GH[W/m^2]"]
        _x0 = _cooling
        _x1 = _heating
        _x2 = _heating
        _y0 = loadshapes.loc[(county,"SFD",_cooling.index)]["electric-cooling[kW/sf]"]
        _y1 = loadshapes.loc[(county,"SFD",_heating.index)]["electric-heating[kW/sf]"]
        _y2 = loadshapes.loc[(county,"SFD",_heating.index)]["electric-supplemental-heating[kW/sf]"]
        _f0 = np.polyfit(_x0.to_list(),_y0.to_list(),1)
        _f1 = np.polyfit(_x1.to_list(),_y1.to_list(),1)
        _f2 = np.polyfit(_x2.to_list(),_y2.to_list(),1)
        break
    plt.plot(_x0,_y0,'.b',label='Cooling')
    plt.plot(_x1,_y1,'.g',label='Heating')
    plt.plot(_x2,_y2,'.r',label='Supplemental')
    plt.plot([Tcool,40],np.polyval(_f0,[Tcool,40]),'-b')
    plt.plot([0,Theat],np.polyval(_f1,[0,Theat]),'-h')
    plt.plot([0,Theat],np.polyval(_f1,[0,Theat]),'-h')
    plt.plot([0,Theat],np.polyval(_f2,[0,Theat]),'-h')
    plt.gca()
    return


@app.cell
def __():
    import marimo as mo
    import os, sys
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    pd.options.display.max_columns = None
    pd.options.display.width = 1024
    return mo, np, os, pd, plt, sys


if __name__ == "__main__":
    app.run()
