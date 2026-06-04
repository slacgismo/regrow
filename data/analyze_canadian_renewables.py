import marimo

__generated_with = "0.17.8"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import utils
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import folium
    return folium, utils


@app.cell
def _(utils):
    canadian_renewables = utils.load_canadian_renewables_data()
    canadian_renewables
    return (canadian_renewables,)


@app.cell
def _(canadian_renewables, folium, reduced_network):
    m2 = folium.Map(location=[52.5, -120.5], zoom_start=5, tiles="OpenStreetMap")
    folium.TileLayer("Esri.WorldImagery").add_to(m2)
    for _id in canadian_renewables[canadian_renewables['Technology'] == 'Solar'].index:
        _lat, _lon = reduced_network.loc[_id][["Lat", "Long"]].values
        _bn = reduced_network.loc[_id]["Bus  Name"][0]
        folium.CircleMarker(
            location=[_lat, _lon], popup=f"<b>{_id}, “node: {_bn}”</b>", radius=4, fill_color="gray", 
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
