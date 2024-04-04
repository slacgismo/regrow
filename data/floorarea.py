import marimo

__generated_with = "0.3.9"
app = marimo.App(width="full")


@app.cell
def __(os, pd, valid):
    #
    # Load building floorarea metadata
    #
    if not os.path.exists("floorarea.csv"):
        if not os.path.exists("metadata.csv.zip"):
            meta = pd.read_parquet("https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2021/resstock_amy2018_release_1/metadata/metadata.parquet")
            meta.to_csv("metadata.csv.zip",index=False,header=True,compression="zip")

        meta = pd.read_csv("metadata.csv.zip",
                           usecols=["in.county","in.geometry_building_type_recs","in.sqft"],
                           index_col=["in.county"],
                          )
        meta.drop(meta[~meta.index.isin(valid)].index,inplace=True)

        restype = {"Multi-Family with 2 - 4 Units" : "MFS",
                   "Multi-Family with 5+ Units" : "MFL",
                   "Single-Family Attached" : "SFA",
                   "Single-Family Detached" : "SFD",
                   "Mobile Home" : "MH",
                  }
        meta["building_type"] = [restype[x] for x in meta["in.geometry_building_type_recs"]]
        meta.drop("in.geometry_building_type_recs",axis=1,inplace=True)

        meta.reset_index(inplace=True)
        meta.set_index(["in.county","building_type"],inplace=True)
    else:
        meta = pd.read_csv("floorarea.csv",index_col=["in.county","building_type"])
    return meta, restype


@app.cell
def __(pd):
    #
    # Load building loadshape data
    #
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
    load = pd.read_csv("loadshapes.csv.zip")
    load["in.county"] = [pumas[x] for x in load.county]
    load.reset_index(inplace=True)
    load.set_index(["in.county","building_type"],inplace=True)
    load.sort_index(inplace=True)
    valid = load.index.get_level_values(0).unique()
    return load, pumas, valid


@app.cell
def __():
    import os, sys
    import marimo as mo
    import pandas as pd
    pd.options.display.max_columns=None
    return mo, os, pd, sys


if __name__ == "__main__":
    app.run()
