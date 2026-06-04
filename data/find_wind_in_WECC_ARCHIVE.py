import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    This notebook contained the process for finding which solar PV generators in the [USGS USWTDB](https://energy.usgs.gov/uswtdb/) are in the WECC service area. The general process is:

    1. Load WECC node GIS data
    2. Load USWTDB system data
    3. Load WECC county information
    4. Find WT systems in WECC counties, by joining on county
    """)
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import os, io, requests, zipfile
    import utils
    import matplotlib.pyplot as plt
    import folium
    return folium, mo, np, pd, plt, utils


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
    ## Load USWTDB
    """)
    return


@app.cell
def _(utils):
    uswtdb = utils.load_uswtdb()
    return (uswtdb,)


@app.cell
def _(uswtdb):
    uswtdb
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
def _(mo):
    mo.md(r"""
    ## Find WT systems that are in WECC counties
    """)
    return


@app.cell
def _(pd, uswtdb, wecc_counties):
    wecc_wt_systems = pd.merge(uswtdb, wecc_counties, how='inner', on=['county', 'state'], suffixes=('_gen', '_county'))
    return (wecc_wt_systems,)


@app.cell
def _(wecc_wt_systems):
    wecc_wt_systems
    return


@app.cell
def _(np):
    np.arange(5) + 2018
    return


@app.cell
def _(np, wecc_wt_systems):
    counts = []
    capacities = []
    years = np.arange(5) + 2018
    for _yr in years:
        _count = len(wecc_wt_systems[wecc_wt_systems['year'] <= _yr])
        _cap = np.sum(wecc_wt_systems[wecc_wt_systems['year'] <= _yr]['capacity[MW]']) / 1e3
        counts.append(_count)
        capacities.append(_cap)
    return capacities, counts, years


@app.cell
def _(capacities, counts, plt, years):
    _fig, _ax = plt.subplots(nrows=2, sharex=True, figsize=(9, 6))
    _ax[0].plot(years, counts, marker='.')
    _ax[1].plot(years, capacities, marker='.')
    _ax[0].set_title('system count, WT')
    _ax[1].set_title('total capacity, WT [GW]')
    _ax[1].set_xticks(years)
    _fig
    return


@app.cell
def _(wecc_wt_systems):
    wecc_wt_systems.to_csv('wecc_wt_systems.csv')
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Assign to nodes
    """)
    return


@app.cell
def _(latlon_list, nodes, utils, wecc_wt_systems):
    assigned_node_hashes = []
    for _ix,_row in wecc_wt_systems.iterrows():
        _best_ix, _latlon, _dist = utils.nearest2((_row['latitude_gen'], _row['longitude_gen']), latlon_list)
        assigned_node_hashes.append(nodes.iloc[_best_ix].name)
    return (assigned_node_hashes,)


@app.cell
def _(assigned_node_hashes):
    len(set(assigned_node_hashes))
    return


@app.cell
def _(assigned_node_hashes, wecc_wt_systems):
    wecc_wt_assigned = wecc_wt_systems.copy()
    wecc_wt_assigned['node geohash'] = assigned_node_hashes
    return (wecc_wt_assigned,)


@app.cell
def _(wecc_wt_assigned):
    wecc_wt_assigned
    return


@app.cell
def _(utils):
    reduced_network = utils.load_reduced_network()
    return (reduced_network,)


@app.cell
def _(folium, reduced_network, wecc_wt_assigned):
    m = folium.Map(location=[37.166, -119.449], zoom_start=5, tiles="OpenStreetMap")
    folium.TileLayer("Esri.WorldImagery").add_to(m)
    for _id in reduced_network.index:
        _lat, _lon = reduced_network.loc[_id][["Lat", "Long"]].values
        _bn = reduced_network.loc[_id]["Bus  Name"][0]
        folium.Marker(
            location=[_lat, _lon], popup=f"<b>{_id}, “node: {_bn}”</b>", icon=folium.Icon(color='gray', icon='map-pin')
        ).add_to(m)
    for _id in set(wecc_wt_assigned['node geohash']):
        _lat, _lon = reduced_network.loc[_id][["Lat", "Long"]].values
        _bn = reduced_network.loc[_id]["Bus  Name"][0]
        folium.Marker(
            location=[_lat, _lon], popup=f"<b>{_id}, “WT: {_bn}”</b>", icon=folium.Icon(color='blue', icon='map-pin')
        ).add_to(m)
    folium.LayerControl().add_to(m);
    return (m,)


@app.cell
def _(m):
    m
    return


if __name__ == "__main__":
    app.run()
