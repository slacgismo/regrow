import marimo

__generated_with = "0.4.7"
app = marimo.App(width="full")


@app.cell
def __(mo):
    mo.md("# Load model validation")
    return


@app.cell
def __(mo):
    set_ui = mo.ui.switch()
    mo.hstack([mo.md("Data aggregation: Node level"),set_ui,mo.md("County level")],justify='start')
    return set_ui,


@app.cell
def __(os, pd, set_ui):
    #
    # Review results of data curation
    #
    folder = "geodata/counties" if set_ui.value else "geodata"
    data = dict([
        (os.path.splitext(os.path.basename(file))[0],
        pd.read_csv(os.path.join(folder if set_ui.value else folder,file),index_col=[0],parse_dates=[0])) for file in os.listdir(folder) if file.endswith(".csv")])

    label = {
        "baseload" : "MW",
        "total" : "MW",
        "heating" : "MW",
        "cooling" : "MW",
        "temperature"  : "$^o$C",
        "wind" : "m/s",
        "solar" : "W/m$^2$",
    }
    return data, folder, label


@app.cell
def __(data, panel_ui, pd, set_ui, utils):
    if set_ui.value:
        _locations = pd.read_csv("counties.csv",usecols=[0,2,5],index_col=[0,1])
        # buslist = list(data[panel_ui.value].columns)
        _locations.index = [' '.join(x) for x in _locations.index]
        buslist = _locations[_locations["geocode"].isin(data[panel_ui.value].columns)].to_dict()["geocode"]
    else:
        _locations = pd.read_csv("wecc240_gis.csv",index_col=[0],usecols=[1,11,12])
        _locations["geocode"] = [utils.geohash(x,y,6) for x,y in _locations[["Lat","Long"]].values]
        _locations.drop(["Lat","Long"],inplace=True,axis=1)
        _locations.index = [x.title() for x in _locations.index]
        _locations = _locations[~_locations.index.duplicated(keep='first')]
        # buslist = dict([(x,y) for x,y in _locations["geocode"].sort_index().to_dict().items() if x in data[panel_ui.value].columns])
        # buslist =_locations["geocode"].sort_index().to_dict()
        buslist = list(data[panel_ui.value].columns)
    # buslist
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
                                                title = geocode_ui.selected_key + " " + panel_ui.value.title(),
                                               )
    _fig.figure.tight_layout()
    _fig
    return


@app.cell
def __(data, geocode_ui, mo, panel_ui):
    num_days = len(data[panel_ui.value][geocode_ui.value]) // 24
    start_day = mo.ui.slider(0, num_days, label='start day')
    num_days = mo.ui.slider(1, 30, label='num days')
    mo.hstack([start_day, num_days])
    return num_days, start_day


@app.cell
def __(data, geocode_ui, label, num_days, panel_ui, start_day):
    _fig = data[panel_ui.value][geocode_ui.value].iloc[int(start_day.value*24):int((start_day.value+num_days.value)*24)].plot(marker = '.',
                                                figsize=(10,5),
                                                markersize = 1,
                                                linewidth = 1,
                                                grid = True,
                                                ylabel = label[panel_ui.value],
                                                title = geocode_ui.selected_key + " " + panel_ui.value.title(),
                                               )
    _fig.figure.tight_layout()
    _fig
    return


@app.cell
def __(data, geocode_ui, panel_ui, sns):
    heatmap_data = data[panel_ui.value][geocode_ui.value].values
    heatmap_data = heatmap_data[:8760].reshape((24,-1), order='F')
    sns.heatmap(heatmap_data, cmap='plasma')
    return heatmap_data,


@app.cell
def __():
    import os, sys
    import marimo as mo
    import pandas as pd
    import utils
    import seaborn as sns

    pd.options.display.max_columns = None
    pd.options.display.width = None
    return mo, os, pd, sns, sys, utils


if __name__ == "__main__":
    app.run()
