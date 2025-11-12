import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    This notebook contained the process for finding which solar PV generators in the [USGS USPVDB](https://energy.usgs.gov/uspvdb/) are in the WECC service area. The general process is:

    1. Load WECC node GIS data
    2. Load USPVDB system data
    3. Load WECC county information
    4. Find PV systems in WECC counties, by joining on county
    """)
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import os, io, requests, zipfile
    import utils
    return mo, pd, utils


@app.cell
def _(mo):
    mo.md(r"""
    ## Load WECC node GIS
    """)
    return


@app.cell
def _(utils):
    nodes = utils.load_reduced_network()
    latlon_list = [
        (nodes.iloc[_ix]['Lat'], nodes.iloc[_ix]['Long']) 
        for _ix in range(len(nodes))
    ]
    return latlon_list, nodes


@app.cell
def _(nodes):
    nodes
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Load USPVDB
    """)
    return


@app.cell
def _(utils):
    uspvdb = utils.load_uspvdb()
    return (uspvdb,)


@app.cell
def _(uspvdb):
    uspvdb
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Load WECC county information
    """)
    return


@app.cell
def _(utils):
    wecc_counties = utils.load_wecc_counties()
    return (wecc_counties,)


@app.cell
def _(wecc_counties):
    wecc_counties
    return


@app.cell
def _(uspvdb):
    _ix = 0
    latlon = (uspvdb.iloc[_ix]['latitude'], uspvdb.iloc[_ix]['longitude'])
    return (latlon,)


@app.cell
def _(latlon, latlon_list, utils):
    utils.nearest2(latlon, latlon_list)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Find PV systems that are in WECC counties
    """)
    return


@app.cell
def _(pd, uspvdb, wecc_counties):
    wecc_pv_systems = pd.merge(wecc_counties, uspvdb, how='inner', on=['county', 'state'])
    return (wecc_pv_systems,)


@app.cell
def _(wecc_pv_systems):
    wecc_pv_systems
    return


@app.cell
def _(wecc_pv_systems):
    wecc_pv_systems.to_csv('wecc_pv_systems.csv')
    return


if __name__ == "__main__":
    app.run()
