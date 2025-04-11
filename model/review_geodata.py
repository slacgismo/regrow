import marimo

__generated_with = "0.12.4"
app = marimo.App(width="medium")


@app.cell
def _(daterange_ui, geocode_ui, geodata_ui, mo, timezone_ui, xrange_ui):
    mo.vstack([mo.hstack([geocode_ui,geodata_ui,xrange_ui,],justify='start'),
               mo.hstack([daterange_ui,timezone_ui],justify='start')])
    mo.hstack([geocode_ui,geodata_ui,xrange_ui,daterange_ui,timezone_ui],justify='start')
    return


@app.cell
def _(geodata_ui, plt, xdata, xrange_ui, ydata):
    units = {
        "Base load": "MVA",
        "Heating load": "MVA",
        "Cooling load": "MVA",
        "Total load": "MVA",
        "Wind generation": "MW",
        "Solar generation": "MW",
        "Wind": "m/s",
        "Solar": "W/m$^2$",
        "Temperature": "$^\\circ$C",
    }
    if xrange_ui.value is None:  # timeseries
        _xlabel = "Date/Time"
    else:  # xy plot
        _xlabel = f"{xrange_ui.selected_key} ({units[xrange_ui.selected_key]})"
    plt.figure(figsize=(10, 5))
    plt.plot(xdata, ydata,"-" if xrange_ui.value is None else ".")
    plt.ylabel(f"{geodata_ui.selected_key} ({units[geodata_ui.selected_key]})")
    plt.xlabel(_xlabel)
    plt.grid()
    plt.gca()
    return (units,)


@app.cell
def _(check_geodata, geodata, mo):
    _tabs = {"Summary": check_geodata}
    _tabs.update(geodata.items())
    mo.ui.tabs(_tabs,
              lazy=True)
    return


@app.cell
def _(check_geodata, dt, geodata, mo):
    geocode_ui = mo.ui.dropdown(
        label="Geocode:",
        options=sorted(check_geodata["geocode"].unique()),
        value=check_geodata["geocode"].iloc[0],
    )
    geodata_ui = mo.ui.dropdown(
        label="Y-axis:",
        options=geodata,
        value=list(geodata.keys())[0]
    )
    daterange_ui = mo.ui.date_range(
        label="Date range:",
        start=dt.date(2018,1,1),#geodata_ui.value[geocode_ui.value].index.min().date(),
        stop=dt.date(2023,1,1),#geodata_ui.value[geocode_ui.value].index.max().date(),
    )
    timezone_ui = mo.ui.dropdown(label="Timezone:",
                                 options={
                                     "UTC": dt.timezone(dt.timedelta(0)),
                                     "PST": dt.timezone(dt.timedelta(hours=-8)),
                                     "PDT": dt.timezone(dt.timedelta(hours=-7)),
                                     "MST": dt.timezone(dt.timedelta(hours=-7)),
                                     "MDT": dt.timezone(dt.timedelta(hours=-6)),
                                    },
                                 value="UTC",
                                 allow_select_none=False,
                                )
    # _xrange = {x:y for x,y in geodata_ui.options.items() if x != geodata_ui.selected_key}
    xrange_ui = mo.ui.dropdown(label="X-axis",options=geodata,value=None)
    return daterange_ui, geocode_ui, geodata_ui, timezone_ui, xrange_ui


@app.cell
def _(daterange_ui, dt, geocode_ui, geodata_ui, timezone_ui, xrange_ui):
    _UTC = dt.timezone(dt.timedelta(0))
    _start = dt.datetime.combine(daterange_ui.value[0],dt.time(0,0,0),tzinfo=_UTC)
    _stop = dt.datetime.combine(daterange_ui.value[1],dt.time(0,0,0),tzinfo=_UTC)
    _data = geodata_ui.value[geocode_ui.value].loc[_start:_stop].resample("1h").mean().sort_index()
    ydata = _data
    if xrange_ui.value is None: # timeseries
        xdata = _data.index.tz_convert(timezone_ui.value)
    else: # xy plot
        xdata = xrange_ui.value[geocode_ui.value].loc[_start:_stop].resample("1h").mean().sort_index()
        _start = max(xdata.index[0],ydata.index[0])
        _stop = min(xdata.index[-1],ydata.index[-1])
        xdata = xrange_ui.value[geocode_ui.value].loc[_start:_stop].resample("1h").mean().sort_index()
        ydata = geodata_ui.value[geocode_ui.value].loc[_start:_stop].resample("1h").mean().sort_index()
    return xdata, ydata


@app.cell
def _(os, pd):
    check_geodata = pd.read_csv("check_geodata.csv")
    _geodata_dir = "../data/geodata"
    _tables = {
        "Base load": "baseload.csv",
        "Heating load": "heating.csv",
        "Cooling load": "cooling.csv",
        "Total load": "total.csv",
        "Wind generation": "wt.csv",
        "Solar generation": "pv.csv",
        "Wind": "wind.csv",
        "Solar": "solar.csv",
        "Temperature": "temperature.csv",
    }
    geodata = {
        x: pd.read_csv(
            os.path.join(_geodata_dir, y), index_col=0, parse_dates=True
        )
        for x, y in _tables.items()
    }
    return check_geodata, geodata


@app.cell
def _():
    import marimo as mo
    import os
    import datetime as dt
    import pandas as pd
    import matplotlib.pyplot as plt
    return dt, mo, os, pd, plt


if __name__ == "__main__":
    app.run()
