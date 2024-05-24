import marimo

__generated_with = "0.6.0"
app = marimo.App()


@app.cell
def __(json):
    with open("model/wecc240.json","r") as fh:
        data = json.load(fh)
    return data, fh


@app.cell
def __(data, pd, px):
    _data = pd.DataFrame(data["objects"]).transpose()
    _data.index.name = "name"
    _data.reset_index(inplace=True)
    if "latitude" in _data and "longitude" in _data:
        _data["latitude"] = [float(x) for x in _data["latitude"]]
        _data["longitude"] = [float(x) for x in _data["longitude"]]
        nodes = _data.loc[
            ~_data["latitude"].isnull() & ~_data["longitude"].isnull()
        ]
        print(nodes)
        map = px.scatter_mapbox(
            nodes,
            lat="latitude",
            lon="longitude",
            hover_name="name",
            text="name",
            zoom=5,
        )
        # map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    else:
        map = None
    map
    return map, nodes


@app.cell
def __():
    import os, sys, json
    import marimo as mo
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    return go, json, mo, os, pd, px, sys


if __name__ == "__main__":
    app.run()
