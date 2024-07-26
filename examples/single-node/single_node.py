import marimo

__generated_with = "0.7.9"
app = marimo.App(width="full")


@app.cell
def __():
    #
    # Simulation environment and defaults
    #
    DATA_SOURCE = "../../data/geodata"
    DATA_FILES = ["temperature","solar","wind","baseload","heating","cooling","total"]
    DATA_GROUPS = {"Weather":["temperature","solar","wind"],"Power":["baseload","heating","cooling","total"]}
    DEFAULT_LOCATION = "9q9wtp"
    return DATA_FILES, DATA_GROUPS, DATA_SOURCE, DEFAULT_LOCATION


@app.cell
def __():
    #
    # Wind performance
    #
    wind_cutin = 5  # m/s
    wind_allin = 10  # m/s
    wind_cutout = 25  # m/s
    wind_rating = 100.0  # MW

    def wind_power(x):
        x[x<=wind_cutin] = 0
        x[x>=wind_cutout] = 0
        x[(x>wind_cutin)&(x<=wind_allin)] = (wind_allin-x)/(wind_allin-wind_cutin)*wind_rating
        x[(x>wind_allin)&(x<wind_cutout)] = wind_rating
        return x
    return wind_allin, wind_cutin, wind_cutout, wind_power, wind_rating


@app.cell
def __():
    #
    # Solar performance
    #
    solar_efficiency = 0.10  # pu
    solar_panelarea = 1000000.0  # m^2

    def solar_power(x):
        return x * solar_efficiency * solar_panelarea / 1e6
    return solar_efficiency, solar_panelarea, solar_power


@app.cell
def __(data, get_location, pd):
    #
    # Generation dispatch
    #
    L = sum([data[x][get_location()] for x in ["baseload","cooling","heating"]])  # loads 
    R = sum([data[x][get_location()] for x in ["solar","wind"]])  # renewables 
    G = pd.DataFrame(data=L-R, index=L.index)
    Q = G.copy() # storage
    Q[get_location()] = 0
    return G, L, Q, R


@app.cell
def __(DATA_SOURCE, os, pd, solar_power, wind_power):
    #
    # Datasets
    #
    datasets = {"temperature": {"converter": lambda x: x*1.8+32, "unit": "$^o$F"},
                "solar": {"converter": solar_power, "unit": "MW"}, 
                "wind": {"converter": wind_power, "unit": "MW"},
                "cooling": {"unit": "MW"},
                "heating": {"unit": "MW"},
                "baseload": {"unit": "MW"},
                "total": {"unit": "MW"},
               }
    data = {}
    for name,spec in datasets.items():
        file = os.path.join(DATA_SOURCE,name+".csv")
        opts = dict(low_memory=True,header=[0],index_col=0,parse_dates=[0])
        data[name] = spec["converter"](pd.read_csv(file,**opts)) if "converter" in spec else pd.read_csv(file,**opts)
    start = min([data[x].index[0] for x in datasets])
    stop = max([data[x].index[-1] for x in datasets])
    return data, datasets, file, name, opts, spec, start, stop


@app.cell
def __(DEFAULT_LOCATION, mo):
    #
    # Location state variable
    #
    get_location, set_location = mo.state(DEFAULT_LOCATION)
    get_day, set_day = mo.state(0)
    get_range, set_range = mo.state(365)
    return (
        get_day,
        get_location,
        get_range,
        set_day,
        set_location,
        set_range,
    )


@app.cell
def __(DATA_SOURCE, os, pd):
    #
    # Location names
    #
    counties = pd.read_csv(os.path.join(DATA_SOURCE,"..","counties.csv"),index_col="geocode")

    return counties,


@app.cell
def __(counties, data, datasets, get_location, mo, set_location, utils):
    #
    # Choose location
    #
    locations = sorted(list(
        set.intersection(*[set(data[x].columns[1:]) for x in datasets])
    ))
    _options = dict([(f"{x} ({counties.loc[utils.nearest(x,counties.index)].county})",x) for x in locations])
    _index = dict([(y,x) for x,y in _options.items()])
    location_ui = mo.ui.dropdown(
        label="Location:",
        on_change=set_location,
        options=_options, # locations,
        value=_index[get_location()],
        allow_select_none=False,
    )
    return location_ui, locations


@app.cell
def __(get_range, mo, set_range):
    #
    # Time range slider
    #
    range_ui = mo.ui.slider(
        steps=[1, 7, 31, 92, 365],
        value=get_range(),
        on_change=set_range,
        show_value=True,
        debounce=True,
    )
    return range_ui,


@app.cell
def __(get_day, get_range, mo, range_ui, set_day, start, stop):
    #
    # Day slider
    #
    _steps = list(range(0,int(stop.as_unit("s").value/3600e9/24) - int(start.as_unit("s").value/3600e9/24),range_ui.value))
    day_ui = mo.md("") if len(_steps)<2 else mo.md(f" (day {mo.ui.slider(
        steps=_steps,
        value=(get_day() // get_range())*get_range(),
        on_change=set_day,
        show_value=True,
        debounce=True,
        # disable = len(_steps)<2,
    )})")
    return day_ui,


@app.cell
def __(get_day, mo, range_ui, set_day, start, stop):
    #
    # Time range navigation button
    #
    def set_prev(_):
        _prev = max(0,get_day()-range_ui.value)
        set_day(_prev)
    def set_next(_):
        _max = int(stop.as_unit("s").value/3600e9/24-range_ui.value) - int(start.as_unit("s").value/3600e9/24)
        _next = min(_max,get_day()+range_ui.value)
        set_day(_next)
    prev_ui = mo.ui.button(label="Back",on_click=set_prev)
    next_ui = mo.ui.button(label="Next",on_click=set_next)
    return next_ui, prev_ui, set_next, set_prev


@app.cell
def __(
    day_ui,
    get_day,
    location_ui,
    mo,
    next_ui,
    pd,
    prev_ui,
    range_ui,
    start,
):
    #
    # Main display
    #
    mo.hstack(
        [
            location_ui,
            mo.md(f"Window by"),
            range_ui,
            mo.md(f"day(s), starting on {start+pd.Timedelta(days=get_day())}"),
            day_ui,
            prev_ui,
            next_ui,
        ],
        justify="start",
    )
    return


@app.cell
def __(
    DATA_GROUPS,
    data,
    datasets,
    get_day,
    get_location,
    get_range,
    mo,
    plt,
):
    _opts = dict(grid=True, figsize=(30, 10), xlabel="Date/Time")
    _tabs = {}
    for _x in DATA_GROUPS["Weather"]:
        _y = data[_x]
        plt.figure()
        _units = datasets[_x]["unit"]
        _tabs[_x.title()] = _y[int(get_day()*24) : int(get_day() + get_range())*24][get_location()].plot(
            ylabel=f"{_x.title()} [{_units}]", **_opts
        )
        _tabs[_x.title()].grid('on',which='minor',axis='x')
        
    mo.vstack([mo.md("# Weather"),mo.ui.tabs(_tabs,lazy=True)])
    return


@app.cell
def __(
    DATA_GROUPS,
    data,
    datasets,
    get_day,
    get_location,
    get_range,
    mo,
    plt,
):
    _opts = dict(grid=True, figsize=(30, 10), xlabel="Date/Time")
    _tabs = {}
    for _x in DATA_GROUPS["Power"]:
        _y = data[_x]
        plt.figure()
        _units = datasets[_x]["unit"]
        _tabs[_x.title()] = _y[int(get_day()*24) : int(get_day() + get_range())*24][get_location()].plot(
            ylabel=f"{_x.title()} [{_units}]", **_opts
        )
        _tabs[_x.title()].grid('on',which='minor',axis='x')
        
    mo.vstack([mo.md("# Load"),mo.ui.tabs(_tabs,lazy=True)])
    return


@app.cell
def __(G, Q, get_day, get_range, mo, plt):
    _opts = dict(grid=True, figsize=(30, 10), xlabel="Date/Time")
    _tabs = {}

    plt.figure()
    _tabs["Generation"] = G[int(get_day()*24) : int(get_day() + get_range()) * 24][G > 0].plot(ylabel=f"Generation dispatch [MW]", label="OK", **_opts)
    plt.plot(G[int(get_day()*24) : int(get_day() + get_range()) * 24][G <= 0], "xr", label="Outage")
    plt.grid('on',which='minor',axis='x')

    plt.figure()
    _tabs["Storage"] = Q[int(get_day()*24) : int(get_day() + get_range()) * 24][G > 0].plot(ylabel=f"Generation dispatch [MW]", label="OK", **_opts)
    plt.grid('on',which='minor',axis='x')

    mo.vstack([mo.md("# Dispatch"),mo.ui.tabs(_tabs,lazy=True)])
    return


@app.cell
def __():
    #
    # Modules
    #
    import os, sys
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from collections.abc import Iterable
    sys.path.append("../../data")
    import requests
    import utils
    return Iterable, mo, np, os, pd, plt, requests, sys, utils


if __name__ == "__main__":
    app.run()
