

import marimo

__generated_with = "0.13.3"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""This notebook reads the county level geodata for heating and cooling loads from the files in `geodata/counties` and estimates the dynamic response of heating and cooling loads to weather.""")
    return


@app.cell
def _(pd, tzinfo):
    # load data
    UTC = tzinfo.TZ("UTC", 0, 0)
    timezones = {
        x: tzinfo.TZ(x, *y)
        for x, y in {
            "EST": (-5, 0),
            "CST": (-6, 0),
            "MST": (-7, 0),
            "PST": (-8, 0),
        }.items()
    }

    _T = pd.read_csv(
        "geodata/counties/temperature.csv",
        index_col=["timestamp"],
        parse_dates=True,
        low_memory=True,
    )
    geocodes = _T.columns

    _C = (
        pd.read_csv(
            "geodata/counties/cooling.csv",
            index_col=["timestamp"],
            parse_dates=True,
            low_memory=True,
        )
        .join(_T, lsuffix="_P", rsuffix="_T")
        .dropna()
    )

    _H = (
        pd.read_csv(
            "geodata/counties/heating.csv",
            index_col=["timestamp"],
            parse_dates=True,
            low_memory=True,
        )
        .join(_T, lsuffix="_P", rsuffix="_T")
        .dropna()
    )

    power = {"heating": _H, "cooling": _C}
    return geocodes, power, timezones


@app.cell
def _(mo):
    K_ui = mo.ui.slider(
        label="Model order:", start=1, stop=50, value=26, show_value=True
    )
    log_ui = mo.ui.checkbox(label="Use log(p)", value=True)
    filter_ui = mo.ui.checkbox(label="Filter", value=True)
    mo.hstack([K_ui, log_ui, filter_ui], justify="start")
    return K_ui, filter_ui, log_ui


@app.cell
def _(K_ui):
    K = K_ui.value # model order
    return (K,)


@app.cell
def _(log_ui, np):
    dolog=log_ui.value
    def f(x):
        return np.log(x+1) if dolog else x
    def g(x):
        return np.exp(x)-1 if dolog else x
    return f, g


@app.cell
def _(K, counties, f, geocodes, mo, np, power, timezones, tzinfo):
    # processing data
    weather_tf = {"heating": {}, "cooling": {}}
    with mo.status.progress_bar(
        title="Processing county heating/cooling data...",
        remove_on_exit=True,
        collection=geocodes,
    ) as _bar:
        for _geocode in geocodes:
            for table, data in power.items():
                _fips = f"{counties.loc[_geocode].fips:05.0f}"
                try:
                    _tz = timezones[tzinfo.TIMEZONES[_fips][:3]]
                except:
                    _tz = timezones[tzinfo.TIMEZONES[_fips[:2]][:3]]
                _X = np.matrix(data[f"{_geocode}_T"]).transpose()
                _Y = f(np.matrix(data[f"{_geocode}_P"]).transpose())
                _L = len(_Y)
                _M = np.hstack(
                    [
                        np.hstack([_Y[n + 1 : _L - K + n] for n in range(K)]),
                        np.hstack([_X[n + 1 : _L - K + n] for n in range(K + 1)]),
                    ]
                )
                _Mt = _M.transpose()
                try:
                    weather_tf[table][_geocode] = np.linalg.solve(
                        _Mt * _M, _Mt * _Y[K + 1 :]
                    )
                except np.linalg.LinAlgError as err:
                    print(
                        f"ERROR [{counties.loc[_geocode].county},{counties.loc[_geocode].usps},{_geocode},{_tz.name}]: {err} ({table} energy={data[f'{_geocode}_P'].sum() / 1000:.1f} GWh) --> x=[[0]*{2 * K + 1}]'",
                        flush=True,
                    )
                    weather_tf[table][_geocode] = np.matrix(
                        np.zeros((2 * K + 1, 1))
                    )
            _bar.update()
    # print(_M)
    return (weather_tf,)


@app.cell
def _(mo, pd):
    counties = pd.read_csv("counties.csv", index_col=["geocode"])
    state_ui = mo.ui.dropdown(label="State:",options=counties.usps.unique(),value=counties.usps.unique()[0])
    return counties, state_ui


@app.cell
def _(counties, mo, state_ui):
    _counties = counties[counties.usps == state_ui.value]
    _options = dict(zip(_counties.county, _counties.index))
    county_ui = mo.ui.dropdown(
        label="County:", options=_options, value=_counties.county.iloc[0]
    )
    return (county_ui,)


@app.cell
def _(mo, power):
    weather_ui = mo.ui.dropdown(label="Weather:",options=power.keys(),value=list(power.keys())[0])
    return (weather_ui,)


@app.cell
def _(county_ui, mo, state_ui, weather_ui):
    mo.hstack([state_ui, county_ui, weather_ui], justify="start")
    return


@app.cell
def _(county_ui, mo, power, state_ui, weather_ui):
    _data = power[weather_ui.value]
    mo.stop(
        not county_ui.value + "_T" in _data,
        f"Data for {county_ui.selected_key} {state_ui.value} ({county_ui.value}) not in weather_tf results",
    )
    week_ui = mo.ui.slider(
        label="Week:",
        start=0,
        stop=int(len(_data[county_ui.value + "_T"]) / 7 / 24),
        value=0 if weather_ui.value == "heating" else 26,
        show_value=True,
    )
    days_ui = mo.ui.slider(
        label="Days:",
        steps=[7, 14, 21, 28, 92, 184, 365],
        value=28,
        show_value=True,
    )
    mo.hstack([week_ui, days_ui], justify="start")
    return days_ui, week_ui


@app.cell
def _(
    K,
    counties,
    county_ui,
    days_ui,
    f,
    filter_ui,
    g,
    mo,
    np,
    plt,
    power,
    weather_tf,
    weather_ui,
    week_ui,
):
    _geocode = county_ui.value
    _data = power[weather_ui.value]
    mo.stop(
        _geocode + "_T" not in _data.columns,
        mo.md("ERROR: county not found in heating/cooling data"),
    )
    _X = np.matrix(_data[f"{_geocode}_T"]).transpose()
    _Y = np.matrix(_data[f"{_geocode}_P"]).transpose()
    _L = len(_Y)
    _M = np.hstack(
        [
            np.hstack([f(_Y[n : _L - K + n - 1]) for n in range(K)]),
            np.hstack([_X[n : _L - K + n - 1] for n in range(K + 1)]),
        ]
    )
    _x = weather_tf[weather_ui.value][_geocode]

    _week = week_ui.value
    _days = days_ui.value
    _start = max(1, 24 * 7 * _week)
    _stop = min(24 * (7 * _week + _days), len(_data[_geocode + "_T"] - 1))
    _window = np.s_[_start:_stop]
    _last = np.s_[(_start - 1) : (_stop - 1)]
    _next = np.s_[(_start + 1) : (_stop + 1)]

    _filter = [0.3, 0.4, 0.3] if filter_ui.value else [0, 1, 0]
    def filter(x, window=False):
        if window:
            return (
                _filter[0] * x[_last]
                + _filter[1] * x[_window]
                + _filter[2] * x[_next]
            )
        else:
            return _filter[0] * x[0:-2] + _filter[1] * x[1:-1] + _filter[2] * x[2:]


    _y = np.full((_L, 1), float("nan"))
    _y[K:-1] = g(_M @ _x)

    plt.figure(figsize=(15, 7))
    plt.plot(
        _data[f"{_geocode}_T"].index[_window],
        filter(_y, True),
        "b",
        label="Model",
    )
    plt.plot(
        _data[f"{_geocode}_T"].index[_window],
        filter(_Y, True),
        "--k",
        label="Data",
    )
    plt.grid()
    plt.ylabel(f"{weather_ui.value.title()} power [MW]")
    plt.legend()
    _err = filter(_y) - filter(_Y)
    plt.title(
        f"{counties.loc[_geocode].county} {counties.loc[_geocode].usps} ({_geocode}): $\\sigma^2$ = {np.nanstd(_err):.2f} MW ({np.nanstd(_err) / np.nanmean(_Y) * 100:.1f}%)"
    )
    P = filter(_y)
    Q = filter(_Y)
    T = _X[1:-1]
    plt.gca()
    return P, Q, T


@app.cell
def _(P, Q, T, plt, weather_ui):
    plt.figure(figsize=(15,15))
    plt.plot(T,P,'.-b',linewidth=0.5,label="Model")
    plt.plot(T,Q,'+:k',linewidth=0.5,label="Data")
    plt.grid()
    plt.ylabel(f"{weather_ui.value.title()} power [MW]")
    plt.xlabel("Temperature [$\\circ$C]")
    plt.gca()
    return


@app.cell
def _():
    import marimo as mo
    import os
    import sys
    import datetime as dt
    import pandas as pd
    import numpy as np
    import scipy as sp
    import tzinfo
    import matplotlib.pyplot as plt
    return mo, np, pd, plt, tzinfo


if __name__ == "__main__":
    app.run()
