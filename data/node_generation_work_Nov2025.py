import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import utils
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from glob import glob
    import folium
    import seaborn
    return folium, mo, pd, utils


@app.cell
def _(mo):
    select_year = mo.ui.switch(label='**select year:** 2011 <> 2018')
    return (select_year,)


@app.cell
def _(pd, utils):
    wecc_gen_2011 = pd.read_csv('wecc240/wecc240raw_generators.csv')
    wecc_gen_2018 = pd.read_excel('wecc240/WECC240_2018_Generation_scheduling.xlsx', sheet_name='Generator', index_col=0)
    network = utils.load_full_network()
    reduced_network = utils.load_reduced_network()
    return network, reduced_network, wecc_gen_2011, wecc_gen_2018


@app.cell
def _(network, pd, wecc_gen_2011, wecc_gen_2018):
    joined_2011 = pd.merge(wecc_gen_2011, network, how='inner', left_on='   I', right_on='Bus  Number')
    joined_2018 = pd.merge(wecc_gen_2018, network, how='inner', left_on='busname', right_on='Bus  Number')
    return joined_2011, joined_2018


@app.cell
def _(joined_2011, joined_2018):
    _grouped = joined_2011[joined_2011['ID'].isin(['DP', 'S'])]\
        .groupby('geohash')
    pv_nodes_2011 = _grouped.first()
    _grouped = joined_2018[joined_2018['Gen_Type'].isin(['PV-1', 'PV-2'])]\
        .groupby('geohash')
    pv_nodes_2018 = _grouped.first()
    return pv_nodes_2011, pv_nodes_2018


@app.cell
def _(joined_2011, joined_2018):
    _grouped = joined_2011[joined_2011['ID'].isin(['W', 'NW', 'SW'])]\
        .groupby('geohash')
    wt_nodes_2011 = _grouped.first()
    _grouped = joined_2018[joined_2018['Gen_Type'].isin(['Wind-1', 'Wind-2'])]\
        .groupby('geohash')
    wt_nodes_2018 = _grouped.first()
    return wt_nodes_2011, wt_nodes_2018


@app.cell
def _(mo, pv_nodes_2011, pv_nodes_2018, wt_nodes_2011, wt_nodes_2018):
    mo.md(f"""
    - {len(set(pv_nodes_2018.index) - set(pv_nodes_2011.index))} PV nodes were added in 2018
    - {len(set(pv_nodes_2011.index) - set(pv_nodes_2018.index))} PV nodes were removed in 2018
    - {len(set(wt_nodes_2018.index) - set(wt_nodes_2011.index))} WT nodes were added in 2018
    - {len(set(wt_nodes_2011.index) - set(wt_nodes_2018.index))} WT nodes were removed in 2018
    """)
    return


@app.cell
def _(
    folium,
    pv_nodes_2011,
    pv_nodes_2018,
    reduced_network,
    select_year,
    wt_nodes_2011,
    wt_nodes_2018,
):
    m = folium.Map(location=[37.166, -119.449], zoom_start=5, tiles="OpenStreetMap")
    folium.TileLayer("Esri.WorldImagery").add_to(m)
    for _id in reduced_network.index:
        _lat, _lon = reduced_network.loc[_id][["Lat", "Long"]].values
        _bn = reduced_network.loc[_id]["Bus  Name"][0]
        folium.CircleMarker(
            location=[_lat, _lon], popup=f"<b>{_id}, “node: {_bn}”</b>", radius=4, fill_color="gray", 
            fill_opacity=0.9, color="black", weight=1
        ).add_to(m)
    if not select_year.value:
        for _id in pv_nodes_2011.index:
            _lat, _lon = reduced_network.loc[_id][["Lat", "Long"]].values
            _bn = reduced_network.loc[_id]["Bus  Name"][0]
            folium.CircleMarker(
            location=[_lat, _lon], popup=f"<b>{_id}, “PV: {_bn}”</b>", radius=4, fill_color="orange", 
            fill_opacity=0.9, color="black", weight=1
        ).add_to(m)
        for _id in wt_nodes_2011.index:
            _lat, _lon = reduced_network.loc[_id][["Lat", "Long"]].values
            _bn = reduced_network.loc[_id]["Bus  Name"][0]
            folium.CircleMarker(
            location=[_lat, _lon], popup=f"<b>{_id}, “WT: {_bn}”</b>", radius=4, fill_color="blue", 
            fill_opacity=0.9, color="black", weight=1
        ).add_to(m)
        for _id in set(wt_nodes_2011.index).intersection(set(pv_nodes_2011.index)):
            _lat, _lon = reduced_network.loc[_id][["Lat", "Long"]].values
            _bn = reduced_network.loc[_id]["Bus  Name"][0]
            folium.CircleMarker(
            location=[_lat, _lon], popup=f"<b>{_id}, “both: {_bn}”</b>", radius=4, fill_color="red", 
            fill_opacity=0.9, color="black", weight=1
        ).add_to(m)
    else:
        for _id in pv_nodes_2018.index:
            _lat, _lon = reduced_network.loc[_id][["Lat", "Long"]].values
            _bn = reduced_network.loc[_id]["Bus  Name"][0]
            folium.CircleMarker(
            location=[_lat, _lon], popup=f"<b>{_id}, “PV: {_bn}”</b>", radius=4, fill_color="orange", 
            fill_opacity=0.9, color="black", weight=1
        ).add_to(m)
        for _id in wt_nodes_2018.index:
            _lat, _lon = reduced_network.loc[_id][["Lat", "Long"]].values
            _bn = reduced_network.loc[_id]["Bus  Name"][0]
            folium.CircleMarker(
            location=[_lat, _lon], popup=f"<b>{_id}, “WT: {_bn}”</b>", radius=4, fill_color="blue", 
            fill_opacity=0.9, color="black", weight=1
        ).add_to(m)
        for _id in set(wt_nodes_2018.index).intersection(set(pv_nodes_2018.index)):
            _lat, _lon = reduced_network.loc[_id][["Lat", "Long"]].values
            _bn = reduced_network.loc[_id]["Bus  Name"][0]
            folium.CircleMarker(
            location=[_lat, _lon], popup=f"<b>{_id}, “both: {_bn}”</b>", radius=4, fill_color="red", 
            fill_opacity=0.9, color="black", weight=1
        ).add_to(m)
    folium.LayerControl().add_to(m);
    return (m,)


@app.cell
def _(m):
    m
    return


@app.cell
def _(select_year):
    select_year
    return


if __name__ == "__main__":
    app.run()
