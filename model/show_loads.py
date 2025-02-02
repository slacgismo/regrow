import marimo

__generated_with = "0.10.17"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(r"""# WECC 240 Loads""")
    return


@app.cell
def _(mo):
    # Control time and place

    get_hour,set_hour = mo.state(0)
    get_location,set_location = mo.state(None)
    return get_hour, get_location, set_hour, set_location


@app.cell
def _(geojson, pd):
    # Load data

    loads = pd.read_csv("loads.csv",
        parse_dates=[0],
        index_col="datetime",
        )
    usmap = geojson.load(open("../data/geojson/usmap.geojson","r"))
    return loads, usmap


@app.cell
def _(loads, np, timestamp_ui, timestamps):
    # Get geocode points

    geocodes = loads.loc[timestamps[timestamp_ui.value]].set_index("geocode")
    points = np.stack(
        [geocodes.longitude.tolist(), geocodes.latitude.tolist()], -1
    )
    return geocodes, points


@app.cell
def _():
    # WIP: map load

    # _loads = loads.loc[loads.index[0]]
    # _points = np.array([_loads.longitude.tolist(), _loads.latitude.tolist()]).transpose()
    # _map = spatial.Voronoi(_points)
    # # spatial.voronoi_plot_2d(_map)
    # plt.show()

    # def area(x,y):
    #     return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
    # _regions = list(zip(_map.points.tolist(),[[tuple(_map.vertices[y]) for y in _map.regions[x]] for x in _map.point_region]))
    # _geocodes = _loads.set_index("geocode")
    # _geocodes = {x:np.unique(_geocodes.loc[x][["longitude","latitude"]].values).tolist() for x in _geocodes.index.unique()}
    # # print(_geocodes)

    return


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
    _options = {"Voltage magnitude":"voltage[pu]","Voltage angle":"voltage[deg]"}
    voltage_ui = mo.ui.dropdown(options=_options,value=list(_options)[0])
    browser_ui = mo.ui.checkbox(label="Show data")
    return (
        addday_ui,
        addhour_ui,
        browser_ui,
        end_ui,
        start_ui,
        subday_ui,
        subhour_ui,
        voltage_ui,
    )


@app.cell
def _(
    addday_ui,
    addhour_ui,
    browser_ui,
    end_ui,
    get_hour,
    mo,
    start_ui,
    subday_ui,
    subhour_ui,
    timestamp_ui,
    timestamps,
    voltage_ui,
):
    # Show UI inputs

    mo.hstack(
        [
            voltage_ui,
            mo.md(str(f"at {timestamps[get_hour()]}")),
            timestamp_ui,
            start_ui,
            subday_ui,
            subhour_ui,
            addhour_ui,
            addday_ui,
            end_ui,
            browser_ui,
        ],
        justify="start",
    )
    return


@app.cell
def _(
    figsize,
    geocodes,
    geojson,
    get_hour,
    interp,
    np,
    plt,
    points,
    timestamps,
    usmap,
    voltage_ui,
):
    # Generate map
    # See https://stackoverflow.com/questions/37872171/how-can-i-perform-two-dimensional-interpolation-using-scipy for a better interpolator

    _values = np.array(geocodes[voltage_ui.value].tolist())
    _x0, _x1 = int(geocodes.longitude.min()-1), int(geocodes.longitude.max())
    _y0, _y1 = int(geocodes.latitude.min()), int(geocodes.latitude.max() + 1)
    _x, _y = np.meshgrid(np.linspace(_x0,_x1,int((_x1-_x0)/0.02)),
                         np.linspace(_y0,_y1,int((_y1-_y0)/0.02)),
                         indexing='xy')
    _z = interp.griddata(points, _values, (_x, _y), method="cubic")

    # draw map
    plt.figure(figsize=figsize)
    plt.imshow(_z, extent=[_x0, _x1, _y1, _y0])

    # draw states
    for _feature in usmap["features"]:
        _lines = np.array(list(geojson.utils.coords(_feature))).T
        # print(_lines)
        plt.plot(*_lines,"k",linewidth=0.5)

    # draw geocoded points
    _index = geocodes.index.unique()
    plt.plot(geocodes.loc[_index].longitude.tolist(),geocodes.loc[_index].latitude.tolist(),'.k')

    # draw empty selection
    highlight, = plt.plot([-105],[34],'oy')

    # finalize map image
    plt.grid()
    plt.xlim([_x0,_x1])
    plt.ylim([_y0,_y1])
    plt.title(f"Voltage angle at {timestamps[get_hour()]}")
    image = plt.gca()
    return highlight, image


@app.cell
def _(data_ui, highlight, np):
    # Update bus selection
    if not data_ui is None:
        selected = np.array(data_ui.value[["longitude","latitude"]].values).transpose().tolist()
        highlight.set_xdata(selected[0])
        highlight.set_ydata(selected[1])
    return (selected,)


@app.cell
def _(browser_ui, geocodes, get_location, mo, set_location):
    # Browse data table
    data_ui = mo.ui.table(geocodes,initial_selection=get_location().values,on_change=set_location) if browser_ui.value else None
    return (data_ui,)


@app.cell
def _(data_ui, image, mo):
    mo.hstack([image,data_ui])
    return


@app.cell
def _():
    # Settings
    figsize=(10,10)
    return (figsize,)


@app.cell
def _():
    # Load modules

    import marimo as mo
    import numpy as np
    import pandas as pd
    import scipy.interpolate as interp
    import scipy.spatial as spatial
    import matplotlib.pyplot as plt
    import geojson
    return geojson, interp, mo, np, pd, plt, spatial


if __name__ == "__main__":
    app.run()
