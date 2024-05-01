import marimo

__generated_with = "0.4.7"
app = marimo.App(width="full")


@app.cell
def __(mo):
    mo.md("# Load model validation")
    return


@app.cell
def __(mo, os):
    #
    # Switch to select network vs county data (if any)
    #
    set_ui = mo.ui.switch() if os.path.exists("geodata/counties") else None
    return set_ui,


@app.cell
def __(
    geocode_ui,
    heatmap_plot,
    local_plot,
    mo,
    num_days,
    panel_ui,
    set_ui,
    start_day,
    system_plot,
    zoom_plot,
):
    #
    # Display
    #
    mo.vstack([
        mo.hstack([mo.md("Data aggregation: Node level"),set_ui,mo.md("County level")],justify='start') if set_ui else mo.md("No county-level data found"),
        panel_ui,
        system_plot,
        geocode_ui,
        mo.ui.tabs({
            "Local" : local_plot,
            "Zoom" : mo.vstack([
                mo.hstack([start_day, num_days],justify='start'),
                [zoom_plot],
            ]),
            "Heatmap": heatmap_plot
        })
    ])
    return


@app.cell
def __(mo):
    #
    # Zoom level state
    #
    get_start,set_start = mo.state(0)
    get_duration,set_duration = mo.state(365)
    return get_duration, get_start, set_duration, set_start


@app.cell
def __(os, pd, set_ui):
    #
    # Data sources
    #
    folder = "geodata/counties" if set_ui and set_ui.value else "geodata"
    data = dict([
        (os.path.splitext(os.path.basename(file))[0],
        pd.read_csv(os.path.join(folder if set_ui and set_ui.value else folder,file),
                    index_col=[0],
                    parse_dates=[0],
                    # date_format="%Y-%m-%d %H:%M:%S+00:00",
                   )) for file in os.listdir(folder) if file.endswith(".csv")])

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
def __(data, panel_ui, pd, set_ui):
    #
    # Network bus list (nodes)
    #
    if set_ui and set_ui.value:
        _locations = pd.read_csv("counties.csv",usecols=[0,2,5],index_col=[0,1])
        # buslist = list(data[panel_ui.value].columns)
        _locations.index = [' '.join(x) for x in _locations.index]
        buslist = _locations[_locations["geocode"].isin(data[panel_ui.value].columns)].to_dict()["geocode"]
    else:
        _locations = pd.read_csv("nodes.csv",index_col=[0])
        buslist = dict([(y.title(),x) for x,y in _locations.sort_values("Bus  Name").to_dict()["Bus  Name"].items() if x in data[panel_ui.value].columns])
        # buslist = list(data[panel_ui.value].columns)
    # buslist
    return buslist,


@app.cell
def __(data, mo):
    #
    # Data panel dropdown
    #
    panel_ui = mo.ui.dropdown(label = "Data panel",
                              options = dict([(x.title(),x) for x in data]),
                              allow_select_none = False,
                              value = list(data.keys())[0].title(),
                             )
    return panel_ui,


@app.cell
def __(data, label, panel_ui):
    #
    # System-level plot
    #
    _data = data[panel_ui.value]
    system_plot = (_data.sum(axis=1)/(1 if label[panel_ui.value].startswith("MW") else len(_data.columns))).plot(marker = '.',
                                                figsize=(10,5),
                                                markersize = 1,
                                                linewidth = 0,
                                                grid = True,
                                                ylabel = label[panel_ui.value],
                                                title = "System-wide " + panel_ui.value.title(),)
    system_plot.figure.tight_layout()
    return system_plot,


@app.cell
def __(buslist, mo):
    #
    # Local level dataset dropdown
    #
    geocode_ui = mo.ui.dropdown(label = "Node name",
                                options = buslist,
                                allow_select_none = False,
                                value = list(buslist)[0],
                               )
    return geocode_ui,


@app.cell
def __(data, geocode_ui, label, panel_ui):
    #
    # Local level plot
    #
    local_plot = data[panel_ui.value][geocode_ui.value].plot(marker = '.',
                                                figsize=(10,5),
                                                markersize = 1,
                                                linewidth = 0,
                                                grid = True,
                                                ylabel = label[panel_ui.value],
                                                title = geocode_ui.selected_key + " " + panel_ui.value.title(),
                                               )
    local_plot.figure.tight_layout()
    return local_plot,


@app.cell
def __(
    data,
    geocode_ui,
    get_duration,
    get_start,
    mo,
    panel_ui,
    set_duration,
    set_start,
):
    #
    # Zoom level controls
    #
    num_days = len(data[panel_ui.value][geocode_ui.value]) // 24
    start_day = mo.ui.slider(0, num_days, label='Start', value=get_start(), on_change=set_start)
    num_days = mo.ui.slider(1, num_days, label='Duration', value=get_duration(), on_change=set_duration)
    return num_days, start_day


@app.cell
def __(data, geocode_ui, label, num_days, panel_ui, start_day):
    #
    # Zoom level plot
    #
    zoom_plot = data[panel_ui.value][geocode_ui.value].iloc[int(start_day.value*24):int((start_day.value+num_days.value)*24)].plot(
        marker = '.',
        figsize=(10,5),
        markersize = 1,
        linewidth = 1,
        grid = True,
        ylabel = label[panel_ui.value],
        title = geocode_ui.selected_key + " " + panel_ui.value.title(),
       )
    zoom_plot.figure.tight_layout()
    return zoom_plot,


@app.cell
def __(data, geocode_ui, label, panel_ui, plt, sns):
    #
    # Heatmap plot
    #
    plt.figure(figsize=(10,5))
    heatmap_data = data[panel_ui.value][geocode_ui.value].values
    heatmap_data = heatmap_data[:8760].reshape((24,-1), order='F')
    heatmap_plot = sns.heatmap(heatmap_data, cmap='plasma')
    heatmap_plot.set(xlabel='Day', ylabel='Hour (UTC)')
    heatmap_plot.collections[0].colorbar.set_label(label[panel_ui.value])
    return heatmap_data, heatmap_plot


@app.cell
def __():
    #
    # Requirements and options
    #
    import os, sys
    import marimo as mo
    import pandas as pd
    import utils
    import states
    import seaborn as sns
    import matplotlib.pyplot as plt
    import datetime as dt

    pd.options.display.max_columns = None
    pd.options.display.width = None
    return dt, mo, os, pd, plt, sns, states, sys, utils


if __name__ == "__main__":
    app.run()
