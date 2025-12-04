import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    This notebook contained the process for finding which solar PV generators in the [USGS USPVDB](https://energy.usgs.gov/uspvdb/) are in the WECC service area, and identifying WECC 240 model nodes that are closest to the generators. The general process is:

    1. Load WECC node GIS data
    2. Load USWTDB system data
    3. Load WECC county information
    4. Find WT systems in WECC counties, by joining on county
    5. Identify the set(s) of nodes from the WECC 240 model that are closest to installed generators for a given year

    ### Notebook output

    - `wecc_wt_systems.csv`: contains a table of USWTDB systems that are in the WECC service area
    - `wt_node_geohashes.txt`: contains a list WECC node geohashes that should be assigned WT generation in our new model
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
    from supervenn import supervenn
    return folium, mo, np, pd, plt, supervenn, utils


@app.cell
def _(mo):
    mo.md(r"""
    ## Load WECC node GIS
    """)
    return


@app.cell
def _(utils):
    nodes = utils.load_reduced_network()
    return (nodes,)


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

    Find closest WECC node to each generator, excluding nodes that are high-voltage and therefore transmission only.
    """)
    return


@app.cell
def _(nodes, np, utils, wecc_wt_systems, wt_nodes_2011, wt_nodes_2018):
    remove_nodes = utils.load_high_voltage_nodes()
    # _ns = nodes
    _ns = nodes.drop(remove_nodes['GEOHASH'])
    latlon_list = [
        (_ns.iloc[_ix]['Lat'], _ns.iloc[_ix]['Long']) 
        for _ix in range(len(_ns))
    ]
    node_sets = {}
    # add in previous models for reference
    node_sets['2011m'] = set(wt_nodes_2011.index)
    node_sets['2018m'] = set(wt_nodes_2018.index)
    for _yr in np.arange(5) + 2018:
        _assigned_node_hashes = []
        for _ix,_row in wecc_wt_systems[wecc_wt_systems['year'] <= _yr].iterrows():
            _best_ix, _latlon, _dist = utils.nearest2((_row['latitude_gen'], _row['longitude_gen']), latlon_list)
            _assigned_node_hashes.append(_ns.iloc[_best_ix].name)
        node_sets[str(_yr)+'d'] = set(_assigned_node_hashes)
    return node_sets, remove_nodes


@app.cell
def _(utils):
    reduced_network = utils.load_reduced_network()
    return (reduced_network,)


@app.cell
def _(folium, node_sets, reduced_network, select_year):
    m = folium.Map(location=[37.166, -119.449], zoom_start=5, tiles="OpenStreetMap")
    folium.TileLayer("Esri.WorldImagery").add_to(m)
    for _id in reduced_network.index:
        _lat, _lon = reduced_network.loc[_id][["Lat", "Long"]].values
        _bn = reduced_network.loc[_id]["Bus  Name"][0]
        folium.CircleMarker(
            location=[_lat, _lon], popup=f"<b>{_id}, “node: {_bn}”</b>", radius=4, fill_color="gray", 
            fill_opacity=0.9, color="black", weight=1
        ).add_to(m)
    for _id in node_sets[str(select_year.value)+'d']:
        _lat, _lon = reduced_network.loc[_id][["Lat", "Long"]].values
        _bn = reduced_network.loc[_id]["Bus  Name"][0]
        folium.CircleMarker(
            location=[_lat, _lon], popup=f"<b>{_id}, “PV: {_bn}”</b>", radius=4, fill_color="orange", 
            fill_opacity=0.9, color="black", weight=1
        ).add_to(m)
    folium.LayerControl().add_to(m);
    return (m,)


@app.cell
def _(mo):
    select_year = mo.ui.slider(start=2018, stop=2022, label='select year')
    return (select_year,)


@app.cell
def _(select_year):
    select_year
    return


@app.cell
def _(node_sets, select_year):
    len(node_sets[str(select_year.value)+'d'])
    return


@app.cell
def _(m):
    m
    return


@app.cell
def _(wt_nodes_2011):
    wt_nodes_2011
    return


@app.cell
def _(pd, utils):
    wecc_gen_2011 = pd.read_csv('wecc240/wecc240raw_generators.csv')
    wecc_gen_2018 = pd.read_excel('wecc240/WECC240_2018_Generation_scheduling.xlsx', sheet_name='Generator', index_col=0)
    network = utils.load_full_network()
    joined_2011 = pd.merge(wecc_gen_2011, network, how='inner', left_on='   I', right_on='Bus  Number')
    joined_2018 = pd.merge(wecc_gen_2018, network, how='inner', left_on='busname', right_on='Bus  Number')
    _grouped = joined_2011[joined_2011['ID'].isin(['W', 'NW', 'SW'])]\
        .groupby('geohash')
    wt_nodes_2011 = _grouped.first()
    _grouped = joined_2018[joined_2018['Gen_Type'].isin(['Wind-1', 'Wind-2'])]\
        .groupby('geohash')
    wt_nodes_2018 = _grouped.first()
    return wt_nodes_2011, wt_nodes_2018


@app.cell
def _(node_sets, plt, supervenn):
    supervenn(list(node_sets.values()), list(node_sets.keys()))
    plt.gcf()
    return


@app.cell
def _(node_sets, np):
    wt_node_geohashes = np.asarray(list(node_sets['2020d']))
    np.savetxt('wt_node_geohashes.txt', wt_node_geohashes, fmt='%s')
    return


@app.cell
def _(remove_nodes):
    remove_nodes
    return


@app.cell
def _(node_sets, remove_nodes):
    node_sets['2022d'].intersection(set(remove_nodes['GEOHASH']))
    return


@app.cell
def _(node_sets):
    node_sets['2018m'] - node_sets['2022d']
    return


@app.cell
def _(folium, node_sets, reduced_network):
    m2 = folium.Map(location=[37.166, -119.449], zoom_start=5, tiles="OpenStreetMap")
    folium.TileLayer("Esri.WorldImagery").add_to(m2)
    for _id in reduced_network.index:
        _lat, _lon = reduced_network.loc[_id][["Lat", "Long"]].values
        _bn = reduced_network.loc[_id]["Bus  Name"][0]
        folium.CircleMarker(
            location=[_lat, _lon], popup=f"<b>{_id}, “node: {_bn}”</b>", radius=4, fill_color="gray", 
            fill_opacity=0.9, color="black", weight=1
        ).add_to(m2)
    for _id in node_sets['2018m'] - node_sets['2022d']:
        _lat, _lon = reduced_network.loc[_id][["Lat", "Long"]].values
        _bn = reduced_network.loc[_id]["Bus  Name"][0]
        folium.CircleMarker(
            location=[_lat, _lon], popup=f"<b>{_id}, “in 2018m: {_bn}”</b>", radius=4, fill_color="orange", 
            fill_opacity=0.9, color="black", weight=1
        ).add_to(m2)
    folium.LayerControl().add_to(m2);
    return (m2,)


@app.cell
def _(m2):
    m2
    return


if __name__ == "__main__":
    app.run()
