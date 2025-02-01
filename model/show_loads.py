import marimo

__generated_with = "0.10.17"
app = marimo.App(width="medium")


@app.cell
def _(geojson, mo, pd):
    # Load data
    loads = pd.read_csv("loads.csv",
        parse_dates=[0],
        index_col="datetime",
        )
    usmap = geojson.load(open("../data/geojson/usmap.geojson","r"))
    get_hour,set_hour = mo.state(0)
    return get_hour, loads, set_hour, usmap


@app.cell
def _(loads, mo, set_hour):
    # Setup UI slider
    timestamps = loads.index.unique().tolist()
    timestamp_ui = mo.ui.slider(
        start=0,
        stop=len(timestamps),
        debounce=True,
        on_change=set_hour,
    )
    return timestamp_ui, timestamps


@app.cell
def _(get_hour, loads, mo, set_hour):
    # Setup UI buttons
    start_ui = mo.ui.button(label="$|\lt$",on_click=lambda x:set_hour(0))
    subday_ui = mo.ui.button(label="$\lt\lt$",on_click=lambda x:set_hour(max(get_hour()-24,0)))
    subhour_ui = mo.ui.button(label="$\lt$",on_click=lambda x:set_hour(max(get_hour()-1,0)))
    addhour_ui = mo.ui.button(label="$\gt$",on_click=lambda x:set_hour(min(get_hour()+1,len(loads.index)-1)))
    addday_ui = mo.ui.button(label="$\gt\gt$",on_click=lambda x:set_hour(min(get_hour()+24,len(loads.index)-1)))
    end_ui = mo.ui.button(label="$\gt|$",on_click=lambda x:set_hour(len(loads.index)-1))
    return addday_ui, addhour_ui, end_ui, start_ui, subday_ui, subhour_ui


@app.cell
def _(
    addday_ui,
    addhour_ui,
    end_ui,
    get_hour,
    mo,
    start_ui,
    subday_ui,
    subhour_ui,
    timestamp_ui,
    timestamps,
):
    # Show UI inputs
    mo.hstack([mo.md(str(timestamps[get_hour()])),timestamp_ui,start_ui,subday_ui,subhour_ui,addhour_ui,addday_ui,end_ui],justify='start')
    return


@app.cell
def _(
    geojson,
    get_hour,
    interp,
    loads,
    np,
    plt,
    timestamp_ui,
    timestamps,
    usmap,
):
    # Generate map
    #
    _loads = loads.loc[timestamps[timestamp_ui.value]].set_index("geocode")
    _points = np.stack([_loads.longitude.tolist(), _loads.latitude.tolist()], -1)
    _values = np.array(_loads["voltage[deg]"].tolist())
    _x0, _x1 = int(_loads.longitude.min()-1), int(_loads.longitude.max())
    _y0, _y1 = int(_loads.latitude.min()), int(_loads.latitude.max() + 1)
    _x, _y = np.meshgrid(np.linspace(_x0,_x1,int((_x1-_x0)/0.02)),
                         np.linspace(_y0,_y1,int((_y1-_y0)/0.02)),
                         indexing='xy')
    _z = interp.griddata(_points, _values, (_x, _y), method="cubic")
    plt.imshow(_z, extent=[_x0, _x1, _y1, _y0])

    # draw states
    for _feature in usmap["features"]:
        _lines = list(np.array(list(geojson.utils.coords(_feature))).T)
        plt.plot(*_lines,"k")

    # draw geocoded points
    _geocodes = _loads.index.unique()
    plt.plot(_loads.loc[_geocodes].longitude.tolist(),_loads.loc[_geocodes].latitude.tolist(),'.k')

    # finalize map image
    plt.grid()
    plt.xlim([_x0,_x1])
    plt.ylim([_y0,_y1])
    plt.title(f"Voltage angle at {timestamps[get_hour()]}")
    return


@app.cell
def _(loads):
    # Browse data table
    loads
    return


@app.cell
def _():
    # Load modules
    import marimo as mo
    import numpy as np
    import pandas as pd
    import scipy.interpolate as interp
    import matplotlib.pyplot as plt
    import geojson
    return geojson, interp, mo, np, pd, plt


if __name__ == "__main__":
    app.run()
