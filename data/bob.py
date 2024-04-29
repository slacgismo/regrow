import marimo

__generated_with = "0.4.7"
app = marimo.App(width="full")


@app.cell
def __(os, pd):
    #
    # Review results of data curation
    #
    data = dict([
        (os.path.splitext(os.path.basename(file))[0],
        pd.read_csv(os.path.join("geodata",file),index_col=[0],parse_dates=[0])) for file in os.listdir("geodata") if file.endswith(".csv")])

    label = {
        "baseload" : "MW",
        "total" : "MW",
        "heating" : "MW/$^o$C",
        "cooling" : "MW/$^o$C",
        "temperature"  : "$^o$C",
        "wind" : "m/s",
        "solar" : "W/m$^2$",
    }
    return data, label


@app.cell
def __(data, panel_ui, pd, utils):
    _wecc = pd.read_csv("wecc240_gis.csv",index_col=[0],usecols=[1,11,12])
    _wecc["geocode"] = [utils.geohash(x,y,6) for x,y in _wecc[["Lat","Long"]].values]
    _wecc.drop(["Lat","Long"],inplace=True,axis=1)
    _wecc.index = [x.title() for x in _wecc.index]
    _wecc = _wecc[~_wecc.index.duplicated(keep='first')]

    #buslist = _wecc["geocode"].sort_index().to_dict()
    buslist = list(data[panel_ui.value].columns)
    return buslist,


@app.cell
def __(data, mo):
    panel_ui = mo.ui.dropdown(label = "Data panel",
                              options = dict([(x.title(),x) for x in data]),
                              allow_select_none = False,
                              value = list(data.keys())[0].title(),
                             )
    panel_ui
    return panel_ui,


@app.cell
def __(data, label, panel_ui):
    _data = data[panel_ui.value]
    _fig = (_data.sum(axis=1)/(1 if label[panel_ui.value].startswith("MW") else len(_data.columns))).plot(marker = '.',
                                                figsize=(10,5),
                                                markersize = 1,
                                                linewidth = 0,
                                                grid = True,
                                                ylabel = label[panel_ui.value],
                                                title = "System-wide " + panel_ui.value.title(),)
    _fig.figure.tight_layout()
    _fig
    return


@app.cell
def __(buslist, mo):
    geocode_ui = mo.ui.dropdown(label = "Bus name",
                                options = buslist,
                                allow_select_none = False,
                                value = list(buslist)[0],
                               )
    geocode_ui
    return geocode_ui,


@app.cell
def __(data, geocode_ui, label, panel_ui):
    _fig = data[panel_ui.value][geocode_ui.value].plot(marker = '.',
                                                figsize=(10,5),
                                                markersize = 1,
                                                linewidth = 0,
                                                grid = True,
                                                ylabel = label[panel_ui.value],
                                                title = geocode_ui.value + " " + panel_ui.value.title(),
                                               )
    _fig.figure.tight_layout()
    _fig
    return


@app.cell
def __():
    import os, sys
    import marimo as mo
    import pandas as pd
    import utils

    pd.options.display.max_columns = None
    pd.options.display.width = None
    return mo, os, pd, sys, utils


if __name__ == "__main__":
    app.run()
