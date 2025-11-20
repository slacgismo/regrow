import marimo

__generated_with = "0.17.8"
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
    import matplotlib.pyplot as plt
    import folium
    from supervenn import supervenn
    return folium, io, mo, np, pd, plt, supervenn, utils


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
def _(mo):
    mo.md(r"""
    ## Find PV systems that are in WECC counties
    """)
    return


@app.cell
def _(pd, uspvdb, wecc_counties):
    wecc_pv_systems = pd.merge(uspvdb, wecc_counties, how='inner', on=['county', 'state'], suffixes=('_gen', '_county'))
    return (wecc_pv_systems,)


@app.cell
def _(wecc_pv_systems):
    wecc_pv_systems
    return


@app.cell
def _(np):
    np.arange(5) + 2018
    return


@app.cell
def _(np, wecc_pv_systems):
    counts = []
    capacities = []
    years = np.arange(5) + 2018
    for _yr in years:
        _count = len(wecc_pv_systems[wecc_pv_systems['year'] <= _yr])
        _cap = np.sum(wecc_pv_systems[wecc_pv_systems['year'] <= _yr]['capacity[MW]']) / 1e3
        counts.append(_count)
        capacities.append(_cap)
    return capacities, counts, years


@app.cell
def _(capacities, counts, plt, years):
    _fig, _ax = plt.subplots(nrows=2, sharex=True, figsize=(9, 6))
    _ax[0].plot(years, counts, marker='.')
    _ax[1].plot(years, capacities, marker='.')
    _ax[0].set_title('system count, PV')
    _ax[1].set_title('total capacity, PV [GW]')
    _ax[1].set_xticks(years)
    _fig
    return


@app.cell
def _(wecc_pv_systems):
    wecc_pv_systems.to_csv('wecc_pv_systems.csv')
    return


@app.cell
def _(io, pd):
    csv_str = """GEOHASH,BUS_I,NAME,BUS_TYPE,VOLTAGE,LOAD,GENERATION,GENOK
    9qhsdk,2603,VICTORVL,PQ,500.0,0.0,,0
    9qhsdk,2607,VICTORVL,PQ,287.0,0.0,,0
    9qq5wv,2901,ELDORADO,PQ,500.0,0.0,,0
    9q5zqv,2902,MOHAVE,PQ,500.0,0.0,,0
    9rg8bx,4003,BURNS,PQ,500.0,0.0,,0
    c21g7u,4007,CELILOCA,PQ,500.0,0.0,,0
    c21g7u,4010,CELILO,PQ,230.0,0.0,,0
    9r0vxp,8001,OLINDA,PQ,500.0,0.0,,0"""
    pd.read_csv(io.StringIO(csv_str), header=0)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Assign to nodes
    """)
    return


@app.cell
def _(latlon_list, nodes, select_year, utils, wecc_pv_systems):
    assigned_node_hashes = []
    for _ix,_row in wecc_pv_systems[wecc_pv_systems['year'] <= select_year.value].iterrows():
        _best_ix, _latlon, _dist = utils.nearest2((_row['latitude_gen'], _row['longitude_gen']), latlon_list)
        assigned_node_hashes.append(nodes.iloc[_best_ix].name)
    return (assigned_node_hashes,)


@app.cell
def _(assigned_node_hashes):
    len(set(assigned_node_hashes))
    return


@app.cell
def _(assigned_node_hashes, select_year, wecc_pv_systems):
    wecc_pv_assigned = wecc_pv_systems[wecc_pv_systems['year'] <= select_year.value].copy()
    wecc_pv_assigned['node geohash'] = assigned_node_hashes
    return (wecc_pv_assigned,)


@app.cell
def _(utils):
    reduced_network = utils.load_reduced_network()
    return (reduced_network,)


@app.cell
def _(folium, reduced_network, wecc_pv_assigned):
    m = folium.Map(location=[37.166, -119.449], zoom_start=5, tiles="OpenStreetMap")
    folium.TileLayer("Esri.WorldImagery").add_to(m)
    for _id in reduced_network.index:
        _lat, _lon = reduced_network.loc[_id][["Lat", "Long"]].values
        _bn = reduced_network.loc[_id]["Bus  Name"][0]
        folium.CircleMarker(
            location=[_lat, _lon], popup=f"<b>{_id}, “node: {_bn}”</b>", radius=4, fill_color="gray", 
            fill_opacity=0.9, color="black", weight=1
        ).add_to(m)
    for _id in set(wecc_pv_assigned['node geohash']):
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
def _(assigned_node_hashes):
    len(set(assigned_node_hashes))
    return


@app.cell
def _(m):
    m
    return


@app.cell
def _(pd, utils):
    wecc_gen_2011 = pd.read_csv('wecc240/wecc240raw_generators.csv')
    wecc_gen_2018 = pd.read_excel('wecc240/WECC240_2018_Generation_scheduling.xlsx', sheet_name='Generator', index_col=0)
    network = utils.load_full_network()
    joined_2011 = pd.merge(wecc_gen_2011, network, how='inner', left_on='   I', right_on='Bus  Number')
    joined_2018 = pd.merge(wecc_gen_2018, network, how='inner', left_on='busname', right_on='Bus  Number')
    _grouped = joined_2011[joined_2011['ID'].isin(['DP', 'S'])]\
        .groupby('geohash')
    pv_nodes_2011 = _grouped.first()
    _grouped = joined_2018[joined_2018['Gen_Type'].isin(['PV-1', 'PV-2'])]\
        .groupby('geohash')
    pv_nodes_2018 = _grouped.first()
    return pv_nodes_2011, pv_nodes_2018


@app.cell
def _(pv_nodes_2011):
    pv_nodes_2011
    return


@app.cell
def _(
    latlon_list,
    nodes,
    np,
    pv_nodes_2011,
    pv_nodes_2018,
    utils,
    wecc_pv_systems,
):
    node_sets = {}
    node_sets['2011m'] = set(pv_nodes_2011.index)
    node_sets['2018m'] = set(pv_nodes_2018.index)
    for _yr in np.arange(5) + 2018:
        _assigned_node_hashes = []
        for _ix,_row in wecc_pv_systems[wecc_pv_systems['year'] <= _yr].iterrows():
            _best_ix, _latlon, _dist = utils.nearest2((_row['latitude_gen'], _row['longitude_gen']), latlon_list)
            _assigned_node_hashes.append(nodes.iloc[_best_ix].name)
        node_sets[str(_yr)+'d'] = set(_assigned_node_hashes)
    return (node_sets,)


@app.cell
def _(node_sets, plt, supervenn):
    supervenn(list(node_sets.values()), list(node_sets.keys()))
    plt.gcf()
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
