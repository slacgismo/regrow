import marimo

__generated_with = "0.12.8"
app = marimo.App(width="medium")


@app.cell
def _(
    aggregate_ui,
    load_file,
    load_view,
    mo,
    select_node,
    solar_file,
    solar_view,
    temperature_file,
    temperature_view,
):
    def _view(file,view):
        return mo.vstack([
            mo.hstack([file,aggregate_ui,select_node],justify='start'),
            view,
        ])
    mo.ui.tabs({
        "Load": _view(load_file,load_view),
        "Temperature": _view(temperature_file,temperature_view),
        "Solar": _view(solar_file,solar_view),
    },
              lazy=True)
    return


@app.cell
def _(mo):
    # Load file
    load_file = mo.ui.file(filetypes=[".csv"],
                                  label="Total load data (CSV):",
                                  # kind="area",
                                 )
    return (load_file,)


@app.cell
def _(get_geodata, load_file):
    # Load data
    load_data = get_geodata(load_file)
    return (load_data,)


@app.cell
def _(load_data, select_data, select_view):
    # Load view
    _data = select_data(load_data)
    load_view = None if _data is None or len(_data.columns) == 0 else select_view(_data)
    return (load_view,)


@app.cell
def _(mo):
    # Temperature file
    temperature_file = mo.ui.file(filetypes=[".csv"],
                                  label="Temperature data (CSV):",
                                  # kind="area",
                                 )
    return (temperature_file,)


@app.cell
def _(get_geodata, temperature_file):
    # Temperature data
    temperature_data = get_geodata(temperature_file)
    return (temperature_data,)


@app.cell
def _(select_data, select_view, temperature_data):
    # Temperature view
    _data = select_data(temperature_data)
    temperature_view = None if _data is None or len(_data.columns) == 0 else select_view(_data)
    return (temperature_view,)


@app.cell
def _(mo):
    # Solar file
    solar_file = mo.ui.file(filetypes=[".csv"],
                                  label="Solar data (CSV):",
                                  # kind="area",
                                 )
    return (solar_file,)


@app.cell
def _(get_geodata, solar_file):
    # Solar data
    solar_data = get_geodata(solar_file)
    return (solar_data,)


@app.cell
def _(select_data, select_view, solar_data):
    # Solar view
    _data = select_data(solar_data)
    solar_view = None if _data is None or len(_data.columns) == 0 else select_view(_data)
    return (solar_view,)


@app.cell
def _(load_data, mo):
    # Node selection
    select_node = mo.ui.multiselect(label="Locations:",
                                    options=[] if load_data is None else load_data.columns,
                                    value=[],
                                   )
    return (select_node,)


@app.cell
def _(mo):
    # Aggregation
    aggregate_ui = mo.ui.dropdown(label="Aggregation:",
                                  options={
                                      "None":lambda x: x.sort_index().round(1),
                                      "Sum":lambda x: x.sum(axis=1).round(1),
                                      "Average":lambda x: x.mean(axis=1).round(1),
                                      "Standard deviation": lambda x: x.std(axis=1).round(1),
                                      "Fractional deviation": lambda x: (x.std(axis=1)/x.mean(axis=1)).round(1),
                                  },
                                  value="None",
                                 )
    return (aggregate_ui,)


@app.cell
def _(io, pd):
    def get_geodata(file_ui):
        return (
            pd.read_csv(io.StringIO(file_ui.contents(0).decode("utf-8")),index_col=0,parse_dates=True)
            if file_ui.contents(0)
            else None
        )
    return (get_geodata,)


@app.cell
def _(aggregate_ui, pd, select_node):
    # Select data
    def select_data(data):
        if data is None:
            return None
        if select_node.value is None:
            return aggregate_ui.value(data)
        return pd.DataFrame(aggregate_ui.value(data[select_node.value]))
    return (select_data,)


@app.cell
def _(mo):
    def select_view(data):
        return mo.ui.tabs(
        {
            "Data": data,
            "Plot": mo.mpl.interactive(data.plot(figsize=(10,5),grid=True)),
            "Map": None,
        },
        lazy=True,
    )
    return (select_view,)


@app.cell
def _():
    import io
    import math
    import marimo as mo
    import numpy as np
    import pandas as pd
    return io, math, mo, np, pd


if __name__ == "__main__":
    app.run()
