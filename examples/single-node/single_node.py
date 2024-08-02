import marimo

__generated_with = "0.7.14"
app = marimo.App(width="full")


@app.cell
def __():
    #
    # Simulation environment and defaults
    #
    DATA_SOURCE = "data/geodata"
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
def __(
    DATA_SOURCE,
    data,
    datasets,
    get_location,
    mo,
    os,
    pd,
    set_location,
    utils,
):
    #
    # Choose location
    #
    locations = sorted(list(
        set.intersection(*[set(data[x].columns[1:]) for x in datasets])
    ))
    _counties = pd.read_csv(os.path.join(DATA_SOURCE,"..","counties.csv"),index_col="geocode")
    _options = dict([(f"{x} ({_counties.loc[utils.nearest(x,_counties.index)].county})",x) for x in locations])
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
def __(get_location, json, utils):
    #
    # Model data
    #
    with open("model/wecc240.json") as fh:
        _glm = json.load(fh)
        assert(_glm["application"]=="gridlabd")
        assert(_glm["version"]>="4.3.10")
    _objects = _glm["objects"]

    _latlons = dict([(x,(float(y["latitude"]),float(y["longitude"]))) for x,y in _objects.items() if y["class"] in ["bus"]])
    _geocodes = dict([(utils.geohash(*y),x) for x,y in _latlons.items()])
    _nearest = _geocodes[utils.nearest(get_location(),_geocodes.keys())]

    node = [x for x,y in _objects.items() if x == _nearest]
    gens = [x for x,y in _objects.items() if y["class"] == "powerplant" and y["parent"] in node]
    loads = [x for x,y in _objects.items() if y["class"] == "load" and y["parent"] in node]
    lines = [x for x,y in _objects.items() if y["class"] == "branch" and (y["to"] in node or y["from"] in node)]
    model = dict(nodes=dict([(x,_objects[x]) for x in node]),
                 generators=dict([(x,_objects[x]) for x in gens]),
                 loads=dict([(x,_objects[x]) for x in loads]),
                 lines=dict([(x,_objects[x]) for x in lines]),
                )
    node_name = list(model["nodes"].keys())[0] # might have to search for one that has the load
    baseKV = float(model["nodes"][node_name]["baseKV"].split()[0])
    line_cap = max(0,min(999,int(sum([y if y>0 else 1000/baseKV for y in [float(_objects[x]["rateA"].split()[0]) for x in lines]]))))
    gens_cap = max(0,min(999,int(sum([float(_objects[x]["summer_capacity"].split()[0]) for x in gens]))))
    load_cap = max(0,min(999,int(sum([complex(_objects[x]["S"].split()[0]).real for x in loads]))))
    return (
        baseKV,
        fh,
        gens,
        gens_cap,
        line_cap,
        lines,
        load_cap,
        loads,
        model,
        node,
        node_name,
    )


@app.cell
def __(gens_cap, line_cap, load_cap, mo):
    get_load,set_load = mo.state(load_cap)
    get_gens,set_gens = mo.state(gens_cap)
    get_line,set_line = mo.state(line_cap)
    return get_gens, get_line, get_load, set_gens, set_line, set_load


@app.cell
def __(
    gens_cap,
    get_gens,
    get_line,
    get_load,
    load_cap,
    mo,
    set_gens,
    set_line,
    set_load,
):
    load_down_ui = mo.ui.button(label="$-$",on_click=lambda x:set_load(max(0,get_load()-1)))
    gens_down_ui = mo.ui.button(label="$-$",on_click=lambda x:set_gens(max(0,get_gens()-1)))
    line_down_ui = mo.ui.button(label="$-$",on_click=lambda x:set_line(max(0,get_line()-1)))

    load_plus_ui = mo.ui.button(label="$+$",on_click=lambda x:set_load(min(999,get_load()+1)))
    gens_plus_ui = mo.ui.button(label="$+$",on_click=lambda x:set_gens(min(999,get_gens()+1)))
    line_plus_ui = mo.ui.button(label="$+$",on_click=lambda x:set_line(min(999,get_line()+1)))

    load_reset_ui = mo.ui.button(label="Reset",on_click=lambda x:set_load(load_cap))
    gens_reset_ui = mo.ui.button(label="Reset",on_click=lambda x:set_gens(gens_cap))
    line_reset_ui = mo.ui.button(label="Reset",on_click=lambda x:set_line(load_cap))

    load_balance_ui = mo.ui.button(label="Balance",on_click=lambda x:set_load(get_gens()+get_line()))
    gens_balance_ui = mo.ui.button(label="Balance",on_click=lambda x:set_gens(max(0,get_load()-get_line())))
    line_balance_ui = mo.ui.button(label="Balance",on_click=lambda x:set_line(abs(get_gens()-get_load())))
    return (
        gens_balance_ui,
        gens_down_ui,
        gens_plus_ui,
        gens_reset_ui,
        line_balance_ui,
        line_down_ui,
        line_plus_ui,
        line_reset_ui,
        load_balance_ui,
        load_down_ui,
        load_plus_ui,
        load_reset_ui,
    )


@app.cell
def __(get_gens, get_line, get_load, mo, set_gens, set_line, set_load):
    load_ui = mo.ui.slider(value=get_load().real,start=0,stop=999,step=1,on_change=set_load,debounce=True,show_value=True)
    gens_ui = mo.ui.slider(value=get_gens(),start=0,stop=999,step=1,on_change=set_gens,debounce=True,show_value=True)
    line_ui = mo.ui.slider(value=get_line(),start=0,stop=999,step=1,on_change=set_line,debounce=True,show_value=True)
    return gens_ui, line_ui, load_ui


@app.cell
def __(
    gens,
    gens_balance_ui,
    gens_down_ui,
    gens_plus_ui,
    gens_reset_ui,
    gens_ui,
    line_balance_ui,
    line_down_ui,
    line_plus_ui,
    line_reset_ui,
    line_ui,
    lines,
    load_balance_ui,
    load_down_ui,
    load_plus_ui,
    load_reset_ui,
    load_ui,
    loads,
    mo,
    node,
):
    mo.md(f"""## WECC Model Extract

    **Bus name**: {", ".join(node)}

    <table>
    <caption><h3>Resources Capacities</h3></caption>
    <tr><th width=100>Type</th><th colspan=2 width=400>Installed capacity</th><th width=400>Objects</th>
    <tr><th>Loads</th><td>{load_ui} MW</td><td>{load_down_ui} {load_plus_ui} {load_reset_ui} {load_balance_ui}</td><td>{"<br/>".join(loads)}</td></tr>
    <tr><th>Generator</th><td>{gens_ui} MW</td><td>{gens_down_ui} {gens_plus_ui} {gens_reset_ui} {gens_balance_ui}</td><td>{"<br/>".join(gens)}</td></tr>
    <tr><th>Imports</th><td>{line_ui} MW</td><td>{line_down_ui} {line_plus_ui} {line_reset_ui} {line_balance_ui}</td><td>{"<br/>".join(lines)}</td></tr>
    <tr><th>Net capacity</th><td>{gens_ui.value+line_ui.value-load_ui.value:.0f} MW</td><td></td><td></td></tr>
    </table>
    """)
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
    import requests
    import json
    sys.path.insert(0,"data")
    import utils


    return Iterable, json, mo, np, os, pd, plt, requests, sys, utils


@app.cell
def __(mo):
    import git
    repo = git.Repo(".")
    repo.config_reader()
    mo.md(f"Currently using repo branch '`{repo.head.reference.checkout()}`'")
    return git, repo


if __name__ == "__main__":
    app.run()
