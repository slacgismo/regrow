import marimo

__generated_with = "0.6.1-dev29"
app = marimo.App(width="full")


@app.cell
def __(mo):
    mo.md(
        """
        # <a href="https://www.arras.energy/" target=_blank>Arras Energy</a> Model Browser

        Version 5 (21 May 2024)

        Written by <a href="email:dchassin@slac.stanford.edu">David P. Chassin</a>, <a href="https://gismo.slac.stanford.edu/" target=_blank>SLAC National Accelerator Laboratory</a>$^1$
        """
    )
    return


@app.cell
def __():
    #
    # Options
    #
    munge_columns = True  # prevent dots from appear in column names (fixes bug in dataframe explorer)
    return munge_columns,


@app.cell
def __(mo):
    #
    # Upload files to view in the browser (JSON only for now)
    #
    uploads = mo.ui.file(
        filetypes=[".json"],
        multiple=True,
        # kind="area",
        label="Upload models",
    )
    uploads
    return uploads,


@app.cell
def __(mo, os, uploads):
    #
    # Model dropdown
    #
    _names = sorted([os.path.splitext(x.name)[0] for x in uploads.value])
    files = mo.ui.dropdown(
        label="Model name:",
        options=dict([(x, n) for n, x in enumerate(_names)]),
        allow_select_none=True,
        value=_names[0] if len(_names) > 0 else None,
    )
    return files,


@app.cell
def __(files, json, mo, uploads):
    #
    # Model checks
    #
    data = (
        json.loads(uploads.value[files.value].contents)
        if not files.value is None
        else {}
    )

    mo.stop(
        files.value is None,
        mo.md(
            "<font color=blue>To get started upload one or more GridLAB-D JSON files.</font>"
        ),
    )
    _name = uploads.value[files.value].name
    info = mo.stop(
        not type(data) is dict,
        mo.md(
            f"<font color=red>ERROR [{_name}]: JSON file is not a dictionary</font>"
        ),
    )
    for _check in ["application", "version", "modules", "classes", "objects"]:
        info = mo.stop(
            not _check in data,
            mo.md(
                f"<font color=red>ERROR [{_name}]: JSON file does not contain {_check} data</font>"
            ),
        )
    info = mo.stop(
        data["application"] != "gridlabd",
        mo.md(
            f"<font color=red>ERROR [{_name}]: JSON 'application' record is not 'gridlabd'</font>"
        ),
    )

    _contents = uploads.value[files.value].contents
    n_bytes = len(_contents)
    _data = json.loads(_contents)
    n_modules = len(_data["modules"])
    n_objects = len(_data["objects"])
    n_classes = len(_data["classes"])
    info = mo.md(
        f"""GridLAB-D {data["version"]}
    $\\qquad \\boxed{{{n_bytes:,}~\\textrm{{bytes}}}} \\quad 
    \\boxed{{{n_modules:,}~\\textrm{{modules}}}} \\quad 
    \\boxed{{{n_classes:,}~\\textrm{{classes}}}} \\quad 
    \\boxed{{{n_objects:,}~\\textrm{{objects}}}}$"""
    )
    return data, info, n_bytes, n_classes, n_modules, n_objects


@app.cell
def __(files, info, mo, uploads):
    #
    # Model info
    #
    mo.hstack(
        [
            (
                files
                if len(uploads.value) > 1
                else mo.md(f"Model name: **{uploads.value[files.value].name}**")
            ),
            info,
        ],
        justify="start",
    )
    return


@app.cell
def __(data, mo):
    #
    # Classes and header values
    #
    _classes = (
        sorted(list(set([x["class"] for x in data["objects"].values()])))
        if "objects" in data
        else []
    )
    classes = mo.ui.dropdown(
        label="Object class:",
        options=dict([(x.replace("_", " ").title(), x) for x in _classes]),
        allow_select_none=True,
        value=_classes[0].replace("_", " ").title() if len(_classes) > 0 else None,
    )
    header = mo.ui.switch(label="Include header properties")
    return classes, header


@app.cell
def __(classes, data, munge_columns, pd):
    #
    # Object data
    #
    _values = (
        dict(
            [
                (x, y)
                for x, y in data["objects"].items()
                if y["class"] == classes.value
            ]
        )
        if "objects" in data
        else {}
    )
    headers = list(data["header"])
    values = pd.DataFrame(_values).sort_index()
    if munge_columns:
        values.columns = [x.replace(".", "_") for x in values.columns]
    return headers, values


@app.cell
def __(mo):
    #
    # State variables
    #
    get_filter, set_filter = mo.state("")
    get_properties, set_properties = mo.state(None)
    get_rows, set_rows = mo.state(10)
    get_transpose, set_transpose = mo.state(False)
    return (
        get_filter,
        get_properties,
        get_rows,
        get_transpose,
        set_filter,
        set_properties,
        set_rows,
        set_transpose,
    )


@app.cell
def __(
    classes,
    get_filter,
    get_properties,
    get_rows,
    get_transpose,
    header,
    headers,
    mo,
    set_filter,
    set_properties,
    set_rows,
    set_transpose,
    values,
):
    #
    # View control
    #
    rows = mo.ui.text(
        label="Object filter expression:",
        value=get_filter(),
        on_change=set_filter,
        # full_width=True,
    )
    columns = mo.ui.multiselect(
        label="Object properties selected:",
        options=[x for x in values.index if header.value or not x in headers],
        value=get_properties(),
        on_change=set_properties,
    )
    maxrows = mo.ui.number(
        start=1,
        stop=100,
        step=1,
        value=get_rows(),
        on_change=set_rows,
        label="Rows per page:",
        debounce=True,
    )
    transpose = mo.ui.switch(
        label="Transpose data", value=get_transpose(), on_change=set_transpose
    )
    mo.vstack(
        [
            mo.hstack(
                [
                    classes,
                    transpose,
                    maxrows,
                ]
            ),
            mo.hstack(
                [
                    rows,
                    columns,
                    header,
                ]
            ),
        ]
    )
    return columns, maxrows, rows, transpose


@app.cell
def __(
    classes,
    columns,
    data,
    get_mapview,
    header,
    headers,
    maxrows,
    mo,
    munge_columns,
    pd,
    re,
    rows,
    transpose,
):
    #
    # Tabs
    #
    _values = (
        dict(
            [
                (x, y)
                for x, y in data["objects"].items()
                if y["class"] == classes.value
            ]
        )
        if "objects" in data
        else {}
    )
    _data = pd.DataFrame(_values).T.sort_index()
    _data.index.name = "name"
    if munge_columns:
        _data.columns = [x.replace(".", "_") for x in _data.columns]
    if not header.value and len(_data) > 0:
        _data.drop(headers, axis=1, inplace=True, errors="ignore")
    if len(rows.value) > 0:
        _data.drop(
            [x for x in _data.index if not re.match(rows.value, x)],
            axis=0,
            inplace=True,
            errors="ignore",
        )
    if len(columns.value) > 0:
        _data.drop(
            [x for x in _data.columns if not x in columns.value],
            axis=1,
            inplace=True,
            errors="ignore",
        )
    mo.ui.tabs(
        {
            "Table": mo.lazy(
                mo.ui.table(
                    _data.T if transpose.value else _data, page_size=maxrows.value
                ),
                True,
            ),
            "Frame": mo.lazy(mo.ui.dataframe(_data), True),
            "Explorer": mo.lazy(mo.ui.data_explorer(_data), True),
            "Map": mo.lazy(get_mapview()),
        }
    )
    return


@app.cell
def __(data, get_labels, pd, px):
    _data = pd.DataFrame(data["objects"]).transpose()
    _data.index.name = "name"
    _data.reset_index(inplace=True)
    if "latitude" in _data and "longitude" in _data:
        _data["latitude"] = [float(x) for x in _data["latitude"]]
        _data["longitude"] = [float(x) for x in _data["longitude"]]
        nodes = _data.loc[
            ~_data["latitude"].isnull() & ~_data["longitude"].isnull()
        ]

        # lines = _data.loc[~data["from"].isnull() & ~_data["to"].isnull()]

        # Nodes
        map = px.scatter_mapbox(
            nodes,
            lat="latitude",
            lon="longitude",
            # hover_name="name",
            text="name" if get_labels() else None,
            zoom=15,
            # hover_data=dict([(x,True) for x in columns.value]) if columns.value else None,
        )
        map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    else:
        map = None
    map
    return map, nodes


@app.cell
def __(data, mo, pd, px):
    #
    # Mapping
    #
    def load_map(objects, oclass=None):
        data = pd.DataFrame(objects).transpose()
        data.index.name = "name"
        data.reset_index(inplace=True)
        if not oclass is None:
            data.drop(data[data["class"] != oclass].index, inplace=True)
        if "latitude" in data and "longitude" in data:
            data["latitude"] = [float(x) for x in data["latitude"]]
            data["longitude"] = [float(x) for x in data["longitude"]]
            nodes = data.loc[
                ~data["latitude"].isnull() & ~data["longitude"].isnull()
            ]
            lines = data.loc[~data["from"].isnull() & ~data["to"].isnull()]
        else:
            return mo.md("No geodata found")
        # # data.set_index(["class", "name"], inplace=True)
        # # classes = {}
        # # for oclass in data.index.get_level_values(0).unique():
        # #     classes[oclass] = data.loc[oclass].dropna(axis=1, how="all")
        # # print(data)

        # Nodes
        map = px.scatter_mapbox(
            nodes,
            lat="latitude",
            lon="longitude",
            # hover_name="name",
            text="name" if get_labels() else None,
            zoom=15,
            # hover_data=dict([(x,True) for x in columns.value]) if columns.value else None,
        )

        # Lines
        # latlon = nodes.reset_index()[["name", "latitude", "longitude"]].set_index(
        #     "name"
        # )
        # latlon = dict(
        #     [(n, (xy["latitude"], xy["longitude"])) for n, xy in latlon.iterrows()]
        # )
        # valid = [
        #     (n, x, y)
        #     for n, x, y in zip(lines["name"], lines["from"], lines["to"])
        #     if x in latlon and y in latlon
        # ]
        # names = [None] * 3 * len(valid)
        # names[0::3] = [x[0] for x in valid]
        # names[1::3] = [x[0] for x in valid]
        # lats = [None] * 3 * len(valid)
        # lats[0::3] = [latlon[x[1]][0] for x in valid]
        # lats[1::3] = [latlon[x[2]][0] for x in valid]
        # lons = [None] * 3 * len(valid)
        # lons[0::3] = [latlon[x[1]][1] for x in valid]
        # lons[1::3] = [latlon[x[2]][1] for x in valid]
        # map.add_trace(
        #     dict(
        #         hoverinfo="skip",
        #         lat=lats,
        #         lon=lons,
        #         line=dict(color="#636efa"),
        #         mode="lines",
        #         subplot="mapbox",
        #         type="scattermapbox",
        #         showlegend=False,
        #     )
        # )

        # if get_satellite():
        #     map.update_layout(
        #         mapbox_style="white-bg",
        #         mapbox_layers=[
        #             {
        #                 "below": "traces",
        #                 "sourcetype": "raster",
        #                 "sourceattribution": "United States Geological Survey",
        #                 "source": [
        #                     "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
        #                 ],
        #             },
        #         ],
        #     )
        # else:
        #     map.update_layout(mapbox_style="open-street-map")
        map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        set_mapview(map)
        return map

    get_labels, set_labels = mo.state(False)
    get_satellite, set_satellite = mo.state(False)
    get_mapview, set_mapview = mo.state(None)

    def change_labels(x):
        set_labels(x)
        load_map(data["objects"])

    def change_satellite(x):
        set_satellite(x)
        load_map(data["objects"])

    load_map(data["objects"])#, classes.value)
    return (
        change_labels,
        change_satellite,
        get_labels,
        get_mapview,
        get_satellite,
        load_map,
        set_labels,
        set_mapview,
        set_satellite,
    )


@app.cell
def __(mo):
    #
    # Footer
    #
    mo.md(
        """---

    $^1$ SLAC National Accelerator Laboratory is operated by Stanford University for the US Department of Energy (DOE) under Contract DE-AC02-67SF00515. This work was funded by the US DOE Office of Electricity Advanced Grid Modeling Program.

    Copyright &copy; 2024, Regents of the Leland Stanford Junior University
    """
    )
    return


@app.cell
async def __():
    #
    # Requirements
    #
    import os, sys, json, re
    import marimo as mo

    import micropip
    await micropip.install("pandas")
    await micropip.install("plotly")

    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    return go, json, micropip, mo, os, pd, px, re, sys


if __name__ == "__main__":
    app.run()
