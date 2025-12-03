import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import folium
    import utils
    return folium, np, utils


@app.cell
def _(utils):
    reduced_network = utils.load_reduced_network()
    return (reduced_network,)


@app.cell
def _(reduced_network):
    reduced_network
    return


@app.cell
def _(m):
    m
    return


@app.cell
def _(folium, np, reduced_network):
    m = folium.Map(location=[41, -118], zoom_start=5, tiles="Esri.WorldImagery")
    folium.TileLayer("OpenStreetMap").add_to(m)
    for _id in reduced_network.index:
        _lat, _lon = reduced_network.loc[_id][["Lat", "Long"]].values
        _bn = reduced_network.loc[_id]["Bus  Name"][0]
        folium.CircleMarker(
            location=[_lat, _lon], popup=f"<b>{_id}, “node: {_bn}”</b>", radius=4, fill_color="black", 
            fill_opacity=0.9, color="black", weight=1
        ).add_to(m)
    for _id in reduced_network[reduced_network['pv_gen'].values].index:
        _lat, _lon = reduced_network.loc[_id][["Lat", "Long"]].values
        _bn = reduced_network.loc[_id]["Bus  Name"][0]
        folium.CircleMarker(
            location=[_lat, _lon], popup=f"<b>{_id}, “PV: {_bn}”</b>", radius=4, fill_color="orange", 
            fill_opacity=0.9, color="black", weight=1
        ).add_to(m)
    for _id in reduced_network[reduced_network['wt_gen'].values].index:
        _lat, _lon = reduced_network.loc[_id][["Lat", "Long"]].values
        _bn = reduced_network.loc[_id]["Bus  Name"][0]
        folium.CircleMarker(
            location=[_lat, _lon], popup=f"<b>{_id}, “WT: {_bn}”</b>", radius=4, fill_color="blue", 
            fill_opacity=0.9, color="black", weight=1
        ).add_to(m)
    _s = np.logical_and(reduced_network['pv_gen'].values, reduced_network['wt_gen'].values)
    for _id in reduced_network[_s].index:
        _lat, _lon = reduced_network.loc[_id][["Lat", "Long"]].values
        _bn = reduced_network.loc[_id]["Bus  Name"][0]
        folium.CircleMarker(
            location=[_lat, _lon], popup=f"<b>{_id}, “PV&WT: {_bn}”</b>", radius=4, fill_color="green", 
            fill_opacity=0.9, color="black", weight=1
        ).add_to(m)
    for _id in ['c2c10y', 'c2u6xt', '9mtzm4']:
        _lat, _lon = reduced_network.loc[_id][["Lat", "Long"]].values
        _bn = reduced_network.loc[_id]["Bus  Name"][0]
        folium.CircleMarker(
            location=[_lat, _lon], popup=f"<b>{_id}, “foreign: {_bn}”</b>", radius=4, fill_color="yellow", 
            fill_opacity=0.9, color="black", weight=1
        ).add_to(m)
    folium.LayerControl().add_to(m);
    return (m,)


if __name__ == "__main__":
    app.run()
