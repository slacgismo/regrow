import marimo

__generated_with = "0.10.17"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(r"""# WECC System (240 bus model)""")
    return


@app.cell
def _(mo):
    # Control time and place
    maptype = "iwd"
    mapsize = (200, 200)

    get_hour,set_hour = mo.state(0)
    get_location,set_location = mo.state(None)
    return get_hour, get_location, mapsize, maptype, set_hour, set_location


@app.cell
def _(geojson, pd):
    # Load data

    loads = pd.read_csv("loads.csv",
        parse_dates=[0],
        index_col="datetime",
        )
    usmap = geojson.load(open("../data/geojson/usmap.geojson","r"))
    # loads
    return loads, usmap


@app.cell
def _(areas, loads, np, timestamp_ui, timestamps):
    # Get geocode points

    geocodes = loads.loc[timestamps[timestamp_ui.value]].set_index("geocode").sort_index()
    points = np.stack(
        [geocodes.longitude.tolist(), geocodes.latitude.tolist()], -1
    )
    geocodes["area[deg^2]"] = [round(areas[x],3) for x in geocodes.index.values.tolist()]
    geocodes["load_density_log[MVA/deg^2]"] = np.round(np.log(geocodes["power[MVA]"] / geocodes["area[deg^2]"]),3)
    # geocodes
    return geocodes, points


@app.cell
def _(loads, np, spatial):
    # WIP: map load

    _loads = loads.loc[loads.index[0]]
    _points = np.array([_loads.longitude.tolist(), _loads.latitude.tolist()]).transpose()
    _map = spatial.Voronoi(_points)
    # spatial.voronoi_plot_2d(_map)
    # plt.show()

    def area(xy):
        x,y = [pt[0] for pt in xy],[pt[1] for pt in xy]
        return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
    _regions = list(zip(_map.points.tolist(),[[tuple(_map.vertices[y]) for y in _map.regions[x]] for x in _map.point_region]))
    _geocodes = _loads.set_index("geocode")
    _geocodes = {x:np.unique(_geocodes.loc[x][["longitude","latitude"]].values).tolist() for x in _geocodes.index.unique()}

    def geocode(xy):
        for gc,pt in _geocodes.items():
            if pt == xy:
                return gc

    areas = {geocode(x):area(y) for x,y in _regions}
    return area, areas, geocode


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
    # Dataset controls

    start_ui = mo.ui.button(label="&#x23EE;", on_click=lambda x: set_hour(0))
    subday_ui = mo.ui.button(
        label="&#x23EA;", on_click=lambda x: set_hour(max(get_hour() - 24, 0))
    )
    subhour_ui = mo.ui.button(
        label="&#x23F4;", on_click=lambda x: set_hour(max(get_hour() - 1, 0))
    )
    addhour_ui = mo.ui.button(
        label="&#x23F5;",
        on_click=lambda x: set_hour(min(get_hour() + 1, len(loads.index) - 1)),
    )
    # pause_ui = mo.ui.button(label="&#x23F8;",disabled=True)
    addday_ui = mo.ui.button(
        label="&#x23E9;",
        on_click=lambda x: set_hour(min(get_hour() + 24, len(loads.index) - 1)),
    )
    end_ui = mo.ui.button(
        label="&#x23ED;", on_click=lambda x: set_hour(len(loads.index) - 1)
    )
    _options = {
        "Voltage magnitude per unit nominal": "voltage[pu]",
        "Voltage angle": "voltage[deg]",
        "Log load density": "load_density_log[MVA/deg^2]",
    }
    dataset_ui = mo.ui.dropdown(options=_options, value=list(_options)[0])
    return (
        addday_ui,
        addhour_ui,
        dataset_ui,
        end_ui,
        start_ui,
        subday_ui,
        subhour_ui,
    )


@app.cell
def _(geocodes, mo):
    # Display controls

    zoom_ui = mo.ui.slider(
        start=1,
        stop=10,
        step=1,
        label="Distance sensitivity:",
        debounce=False,
        value=4,
    )
    browser_ui = mo.ui.checkbox(label="Show")
    stack_ui = mo.ui.checkbox(label="Stack",value=True)
    showall_ui = mo.ui.checkbox(label="All")
    sort_ui = mo.ui.dropdown(label="Sort:", options=geocodes.columns)
    reverse_ui = mo.ui.checkbox(label="Descending")
    return browser_ui, reverse_ui, showall_ui, sort_ui, stack_ui, zoom_ui


@app.cell
def _(
    addday_ui,
    addhour_ui,
    browser_ui,
    dataset_ui,
    end_ui,
    get_hour,
    mo,
    reverse_ui,
    showall_ui,
    sort_ui,
    stack_ui,
    start_ui,
    subday_ui,
    subhour_ui,
    timestamp_ui,
    timestamps,
    zoom_ui,
):
    # Show UI inputs

    (mo.vstack if stack_ui.value else mo.hstack)([
        mo.hstack(
            [
                dataset_ui,
                mo.md(str(f"at {timestamps[get_hour()]}")),
                timestamp_ui,
                start_ui,
                subday_ui,
                subhour_ui,
                # pause_ui,
                addhour_ui,
                addday_ui,
                end_ui,
            ],justify="start"),
        mo.hstack([
                zoom_ui,
                browser_ui,
                stack_ui,
                showall_ui,
                sort_ui,
                reverse_ui,
            ],
            justify="start",
        ),
    ])
    return


@app.cell
def _(
    dataset_ui,
    figsize,
    geocodes,
    geojson,
    get_hour,
    get_map_cubic,
    get_map_idw,
    mapsize,
    maptype,
    mo,
    np,
    plt,
    timestamps,
    usmap,
):
    # Draw map 

    _values = np.array(geocodes[dataset_ui.value].tolist())
    _xx, _yy = np.array(geocodes.longitude.values.tolist()), np.array(geocodes.latitude.values.tolist())

    _x0, _x1 = int(min(_xx) - 1), int(max(_xx))
    _y0, _y1 = int(min(_yy)), int(max(_yy) + 1)
    _xi, _yi = np.linspace(_x0, _x1, mapsize[0]).tolist(),np.linspace(_y0, _y1, mapsize[1]).tolist()

    if maptype == 'int': # cubic interpolation method
        mapdata = get_map_cubic(_xx,_yy,_values,_xi,_yi)
    elif maptype == 'iwd': # inverse distance method
        mapdata = get_map_idw(_xx,_yy,_values,_xi,_yi)
    else:
        mo.stop("invalid map method specified")

    # draw map
    plt.figure(figsize=figsize)
    _ax = plt.imshow(mapdata, extent=[_x0, _x1, _y1, _y0])

    # draw states
    for _feature in usmap["features"]:
        _lines = np.array(list(geojson.utils.coords(_feature))).T
        # print(_lines)
        plt.plot(*_lines, "k", linewidth=0.5)

    # draw geocoded points
    _index = geocodes.index.unique()
    plt.plot(
        geocodes.loc[_index].longitude.tolist(),
        geocodes.loc[_index].latitude.tolist(),
        ".k",
    )

    # draw empty selection
    highlight, = plt.plot([-105], [34], "oy")

    # finalize map image
    plt.grid()
    plt.xlim([_x0, _x1])
    plt.ylim([_y0, _y1])
    plt.title(f"{dataset_ui.selected_key} at {timestamps[get_hour()]}")
    image = plt.gca()
    _cb = _ax.figure.add_axes([0.95,0.165,0.03,0.662])
    plt.colorbar(_ax,cax=_cb)
    None
    return highlight, image, mapdata


@app.cell
def _(interp, mapsize, np, points, zoom_ui):
    # Map generation routines

    def get_map_cubic(_xx,_yy,_values,_xi,_yi):
        _x, _y = np.meshgrid(_xi, _yi, indexing="xy")
        return interp.griddata(points, _values, (_x, _y), method="cubic")

    def get_map_idw(_xx,_yy,_values,_xi,_yi):
        _x, _y = np.meshgrid(_xi, _yi, indexing="xy")
        _pts = np.vstack((_xx, _yy)).T
        _grd = np.vstack([_x.flatten(), _y.flatten()]).T
        _d0 = np.subtract.outer(_pts[:, 0], _grd[:, 0])
        _d1 = np.subtract.outer(_pts[:, 1], _grd[:, 1])
        _weights = np.power(np.hypot(_d0, _d1),-zoom_ui.value)
        _weights /= _weights.sum(axis=0)
        return np.dot(_weights.T, _values).reshape(mapsize)
    return get_map_cubic, get_map_idw


@app.cell
def _(data_ui, geocodes, highlight, np):
    # Update bus selection

    if not data_ui is None and (_selection := data_ui.value.index.values).tolist():
        selected = np.array(geocodes.loc[_selection][["longitude","latitude"]].values.tolist()).transpose().tolist()
        highlight.set_xdata(selected[0])
        highlight.set_ydata(selected[1])
    else:
        highlight.set_xdata([])
        highlight.set_ydata([])
    return (selected,)


@app.cell
def _(
    browser_ui,
    geocodes,
    get_location,
    mo,
    reverse_ui,
    set_location,
    showall_ui,
    sort_ui,
):
    # Browse data table

    try:
        _initial = get_location().reset_index().index.tolist()
    except:
        _initial = None

    if sort_ui.value:
        geocodes.sort_values(sort_ui.value,inplace=True,ascending=not reverse_ui.value)

    data_ui = (
        mo.ui.table(
            geocodes[geocodes.columns if showall_ui.value else geocodes.columns[2:-2]], # drop lat/lon and calculated values
            initial_selection=_initial,
            on_change=set_location,
            page_size=18,
        )
        if browser_ui.value
        else None
    )
    return (data_ui,)


@app.cell
def _(data_ui, image, mo, stack_ui):
    # Display map and data
    (mo.vstack if stack_ui.value else mo.hstack)([mo.hstack([image]), data_ui if not data_ui is None else mo.md("")])
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
