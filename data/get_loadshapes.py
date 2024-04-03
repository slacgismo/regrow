import marimo

__generated_with = "0.3.4"
app = marimo.App(width="full")


@app.cell
def __():
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
def __(pd, pumas, resstock, restype):
    result = []
    for county,puma in pumas.items():
        for building_type in restype.keys():
            print("Processing",county,building_type,"...") 
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
                             "electric-heating","electric-supplemental-heating",
                             "electric-cooling","electric-pv","electric-total",
                             "gas-heating","gas-total",
                             "propane-heating","fueloil-heating","energy-total",
                            ]
            result.append(_data)
    result = pd.concat(result)
    return building_type, county, puma, result


@app.cell
def __(result):
    result.round({"electric-heating":2,
                  "electric-supplemental-heating":2,
                  "electric-cooling":2,
                  "electric-pv":2,
                  "electric-total":2,
                  "gas-heating":2,
                  "gas-total":2,
                  "propane-heating":2,
                  "fueloil-heating":2,
                  "energy-total":2,
                 }).to_csv("loadshapes.csv.zip",index=False,header=True,compression='gzip')
    return


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    return mo, pd


if __name__ == "__main__":
    app.run()
