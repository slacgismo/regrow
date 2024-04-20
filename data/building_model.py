import marimo

__generated_with = "0.3.9"
app = marimo.App(width="full")


@app.cell
def __(pd):
    _columns = {"county" : "county",
                "building_type" : "building_type",
                "datetime" : "datetime",
                "electric-cooling[kWh]" : "cooling",
                "electric-heating[kWh]" : "heating",
                "electric-supplemental-heating[kWh]" : "supplemental",
                "electric-pv[kWh]" : "pv",
                "electric-total[kWh]" : "total",
               }
    data = pd.read_csv("resstock.csv.zip",
                       index_col=["county","building_type","datetime"],
                       parse_dates=["datetime"],
                       usecols = _columns.keys(),
                      ).rename(axis='columns',mapper=_columns).round(1).sort_index()
    data.heating += data.supplemental
    data.drop("supplemental",axis=1,inplace=True)
    return data,


@app.cell
def __(data, pd, plt):
    for county in data.index.get_level_values(0).unique():
        for building_type in data.index.get_level_values(1).unique():
            print(f"Generating loadshapes/{county} {building_type}...",flush=True)
            temp = pd.read_csv(f"weather/CA-{county.replace(' ','_')}.csv.zip",
                               usecols = ["datetime","TD[degC]","GH[W/m^2]"],
                               index_col=["datetime"],
                               parse_dates=["datetime"],
            ).rename(axis='columns',mapper={"datetime":"datetime","TD[degC]":"temperature","GH[W/m^2]":"solar"})
            result = temp.join(data.loc[county,building_type])
            plt.figure(figsize=(20,10))
            plt.plot(result["temperature"]*1.8+32,
                     result["total"],
                     '.')
            plt.grid()
            plt.title(f"{county} {building_type}")
            plt.savefig(f"loadshapes/{county} {building_type} (scatter).png")
            plt.close()
            result.to_csv(f"loadshapes/{county} {building_type}.csv",index=True,header=True)
    return building_type, county, result, temp


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import matplotlib.pyplot as plt
    return mo, pd, plt


if __name__ == "__main__":
    app.run()
