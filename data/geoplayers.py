import marimo

__generated_with = "0.3.9"
app = marimo.App()


@app.cell
def __(pd):
    buildings = pd.read_csv("buildings.csv",
                            index_col = ["State","county","BuildingType"]
                           )
    return buildings,


@app.cell
def __(pd):
    loadshapes = pd.read_csv("loadshapes.csv.zip",
                       index_col = ["county","building_type","datetime"],
                      )
    return loadshapes,


@app.cell
def __(loadshapes):
    loadshapes
    return


@app.cell
def __(buildings):
    buildings
    return


@app.cell
def __():
    import os, sys
    import marimo as mo
    import pandas as pd
    return mo, os, pd, sys


if __name__ == "__main__":
    app.run()
