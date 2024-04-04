import marimo

__generated_with = "0.1.82"
app = marimo.App(width="full")


@app.cell
def __():
    #
    # Configure processing of loadshape data
    #
    pumas = {
        "Alameda" : "g0600010",
        "Contra Costa" : "g0600130",
        "Los Angeles" : "g0600370",
        "Riverside" : "g0600650",
        "San Bernadino" : "g0600710",
        "San Diego" : "g0600730",
        "San Francisco" : "g0600750",
        "San Mateo" : "g0600810",
        "Santa Clara" : "g0600850",
    }
    restype = {"multi-family with 2 - 4 units" : "MFS",
               "multi-family with 5plus units" : "MFL",
               "single-family attached" : "SFA",
               "single-family detached" : "SFD",
               "mobile home" : "MH",
              }
    resstock = "https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2021/resstock_amy2018_release_1/timeseries_aggregates/by_county/state=CA"
    return pumas, resstock, restype


@app.cell
def __(os, pd, pumas, resstock, restype):
    #
    # Download loadshapes
    #
    if not os.path.exists("resstock.csv.zip"):
        result = []
        for county,puma in pumas.items():
            for building_type in restype.keys():
                print("Downloading",county,building_type,"...") 
                _data = pd.read_csv(f"{resstock}/{puma}-{building_type.replace(' ','_')}.csv",
                                    usecols = ["in.county","in.geometry_building_type_recs","timestamp",
                                               "out.electricity.heating.energy_consumption",
                                               "out.electricity.heating_supplement.energy_consumption",
                                               "out.electricity.cooling.energy_consumption",
                                               "out.electricity.pv.energy_consumption",
                                               "out.electricity.total.energy_consumption",
                                               "out.natural_gas.heating.energy_consumption",
                                               "out.natural_gas.total.energy_consumption",
                                               "out.propane.heating.energy_consumption",
                                               "out.fuel_oil.heating.energy_consumption",
                                               "out.site_energy.total.energy_consumption",
                                              ]
                                   )
                _data['in.county'] = county
                _data['in.geometry_building_type_recs'] = restype[building_type]
                _data.columns = ["county","building_type","datetime",
                                 "electric-heating[kWh]","electric-supplemental-heating[kWh]",
                                 "electric-cooling[kWh]","electric-pv[kWh]","electric-total[kWh]",
                                 "gas-heating[kWh]","gas-total[kWh]",
                                 "propane-heating[kWh]","fueloil-heating[kWh]","energy-total[kWh]",
                                ]
                result.append(_data)
        result = pd.concat(result)
        print("Saving resstock data...")
        result.to_csv("restock.csv.zip",index=False,header=True,compression="zip")
    else:
        print("Loading resstock data...")
        result = pd.read_csv("resstock.csv.zip")
    return building_type, county, puma, result


@app.cell
def __(pd, pumas, result):
    # 
    # Normalize to floor area
    #
    print("Normalizing to power intensity...")
    dt = 0.25; # timestep in hours
    _building_data = pd.read_csv("floorarea.csv",index_col=["in.county","building_type"])
    _area = _building_data.groupby(["in.county","building_type"]).sum()
    _area.columns = ["floor_area[sf]"]
    _area.reset_index(inplace=True)
    _pumas = dict([y.upper(),x] for x,y in pumas.items())
    _area["county"] = [_pumas[x] for x in _area["in.county"]]
    _area.drop("in.county",inplace=True,axis=1)
    _area.set_index(["county","building_type"],inplace=True)
    data = result.set_index(["county","building_type"]).join(_area)
    columns = []
    for column in data.columns:
        if column.endswith("[kWh]"):
            data[column] = data[column] / data["floor_area[sf]"] / dt
        columns.append(column.replace("[kWh]","[kW/sf]"))
    data.columns = columns
    data.drop("floor_area[sf]",axis=1,inplace=True)
    data.reset_index(inplace=True)
    data.set_index(["county","building_type","datetime"],inplace=True)
    data.sort_index(inplace=True)
    return column, columns, data, dt


@app.cell
def __(data):
    #
    # Save loadshape data
    #
    print("Saving loadshape data...")
    data.to_csv("loadshapes.csv.zip",index=True,header=True,compression='zip')
    return


@app.cell
def __(data, os, plt, pumas, restype):
    #
    # Generate plots
    #
    for _county in pumas.keys():
        print("Checking for updates to",_county," County...")
        for _name,_type in restype.items():
            _file = f"loadshapes/{_county} {_type}.png"
            if not os.path.exists(_file):
                print("Generating",_file,"...")
                data.loc[(_county,_type)].plot(figsize=(20,8))
                plt.title(f"{_county} County CA - {_name}")
                plt.grid()
                plt.ylabel('Power intensity')
                plt.xlabel('Date')
                plt.savefig(_file)
                plt.close()
    return


@app.cell
def __():
    #
    # Requirements
    #
    import os, sys
    import marimo as mo
    import pandas as pd
    import matplotlib.pyplot as plt
    return mo, os, pd, plt, sys


if __name__ == "__main__":
    app.run()
