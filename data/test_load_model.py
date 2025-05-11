import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    This notebook tests different load models relative to a simple "bathtub" load model.

    - [x] Read load and weather data
    - [ ] Hold out 5-7 (consecutive/random) days each month (or 60-100 consecutive or random hours?)
    - [ ] Run the original county-level project method
    - [ ] Benchmark the bathtub model for the selected county (RMSE and % error tests) on hold out data
    - [ ] Implement scaling for load growth, upgrades, and electrification.
    - [ ] Implement experimental models

    ## Todo list
    - [ ] Zoom controls on plots to see hourly behavior
    - [ ] NERC model balance temperatures slider
    """
    )
    return


@app.cell
def _(county_ui, fraction_ui, graph_ui, mo, state_ui, wecc_ui):
    mo.hstack([wecc_ui,state_ui,county_ui,graph_ui,fraction_ui],justify='start')
    return


@app.cell
def _(
    comstock_ui,
    errors_ui,
    get_errors,
    mo,
    models_ui,
    resstock_ui,
    sys,
    totals_ui,
    weather_ui,
):
    # main UI
    try:
        result = mo.ui.tabs({
            "Residential" : resstock_ui,
            "Commercial" : comstock_ui,
            "Weather" : weather_ui,
            "Totals" : totals_ui,
            "Models" : models_ui,
            f"Errors ({len(get_errors())})" if get_errors() else "Errors": errors_ui,
            },
            lazy=True)
    except Exception as err:
        e_type,e_value,e_trace = sys.exc_info()
        result = mo.md(f"**EXCEPTION**: {err}")
    result
    return


@app.cell
def _(mo):
    # keep track of state selection when selecting WECC region
    get_state,set_state = mo.state("CA")
    return get_state, set_state


@app.cell
def _(load_models):
    counties = load_models.County._counties
    return (counties,)


@app.cell
def _(counties, get_state, load_models, mo, set_state, wecc_ui):
    # select state
    _options = load_models.County._regions["WECC"] if wecc_ui.value else counties.usps.unique()
    state_ui = mo.ui.dropdown(
        label="State:", options=_options, value=get_state(), on_change=set_state
    )
    return (state_ui,)


@app.cell
def _(mo):
    def add_error(group, message):
        if not message:
            return
        errors = get_errors()
        if group not in errors:
            errors[group] = []
        errors[group].append(message)
        set_errors(errors)

    get_errors, set_errors = mo.state({})
    return add_error, get_errors, set_errors


@app.cell
def _(counties, mo, set_errors, state_ui):
    _options = {
        y.county: x for x, y in counties.iterrows() if y.usps == state_ui.value
    }

    county_ui = mo.ui.dropdown(
        label="County:",
        options=_options,
        on_change=lambda x: set_errors({}),
    )
    return (county_ui,)


@app.cell
def _(mo):
    wecc_ui = mo.ui.checkbox(label="WECC",value=False)
    graph_ui = mo.ui.checkbox(label="Show graph",value=False)
    fraction_ui = mo.ui.checkbox(label="Show fractional load")
    return fraction_ui, graph_ui, wecc_ui


@app.cell
def _(county_ui, load_models, mo):
    mo.stop(county_ui.value==None,mo.md("**HINT**: Select a county"))
    county = load_models.County(county_ui.value)
    weather = load_models.Weather(county).data
    return county, weather


@app.cell
def _(add_error, county, load_models, mo):
    with mo.capture_stderr() as _errors:
        building_loads = {x: {} for x in load_models.Loads._buildings}
        with mo.status.progress_bar(
            title=f"Processing {county}...",
            total=sum([len(x) for x in load_models.Loads._buildings]),
            remove_on_exit=True
        ) as _bar:
            for sector in load_models.Loads._buildings:
                _bar.title = f"Loading {sector} buildings..."
                for building in load_models.Loads._buildings[sector]:
                    _bar.subtitle = f"Reading {building} building data..."
                    try:
                        building_loads[sector][building] = load_models.Loads(
                            county=county, sectors=[sector], buildings=[building]
                        )
                    except Exception as err:
                        add_error(sector,err)
                    _bar.update()
                for _error in _errors.getvalue().split("\n"):
                    add_error(sector, _error)
    return (building_loads,)


@app.cell
def _(TIMEZONES, TZINFO, counties, county_ui, puma):
    # Generate FIPS code and timezone info
    fips = f"{counties.loc[county_ui.value]['fips']:05.0f}"
    if fips[:2] in TIMEZONES:
        tz = TIMEZONES[fips[:2]][:3]
    elif puma in TIMEZONES:
        tz = TIMEZONES[fips][:3]
    else:
        raise Exception(f"unable to find timezone for {fips=})")
    timezone = TZINFO[tz]
    return (timezone,)


@app.cell
def _(building_loads, load_models):
    # residential building data
    resstock = {}
    res_buildings = load_models.Loads._buildings["residential"]
    for _building, _data in building_loads["residential"].items():
        _data.data["baseload[%]"] = _data.data["baseload[MW]"] / _data.data["total[MW]"] * 100
        _data.data["auxheat[%]"] = _data.data["auxheat[MW]"] / _data.data["total[MW]"] * 100
        _data.data["heating[%]"] = _data.data["heating[MW]"] / _data.data["total[MW]"] * 100
        _data.data["cooling[%]"] = _data.data["cooling[MW]"] / _data.data["total[MW]"] * 100
        resstock[_building] = _data.data
    return res_buildings, resstock


@app.cell
def _(building_loads, load_models):
    # commercial building data
    comstock = {}
    com_buildings = load_models.Loads._buildings["commercial"]
    for _building, _data in building_loads["commercial"].items():
        _data.data["baseload[%]"] = _data.data["baseload[MW]"] / _data.data["total[MW]"] * 100
        _data.data["auxheat[%]"] = _data.data["auxheat[MW]"] / _data.data["total[MW]"] * 100
        _data.data["heating[%]"] = _data.data["heating[MW]"] / _data.data["total[MW]"] * 100
        _data.data["cooling[%]"] = _data.data["cooling[MW]"] / _data.data["total[MW]"] * 100
        comstock[_building] = _data.data
    return com_buildings, comstock


@app.cell
def _(
    com_buildings,
    comstock,
    fraction_ui,
    graph_ui,
    mo,
    pd,
    res_buildings,
    resstock,
    timezone,
    weather,
):
    # Generate tab contents
    with mo.status.spinner(
        title="Generating plots" if graph_ui.value else "Updating tables",
        remove_on_exit=True,
    ) as _spinner:
        try:
            resstock_ui = mo.ui.tabs(
                {
                    res_buildings[x]: (
                        y[[f"{z}[{'%' if fraction_ui.value else 'MW'}]" for z in ["baseload", "heating", "cooling"]]]
                        .resample("1d")
                        .mean()
                        .plot.area(
                            figsize=(15, 10),
                            color=["g", "r", "b"],
                            grid=True,
                            ylabel="Load [MW]",
                            xlabel=f"Date/Time ({timezone.name})"
                        )
                        if graph_ui.value
                        else y.round(1)
                    )
                    for x, y in resstock.items() if isinstance(y,pd.DataFrame)
                },
                lazy=True,
            )
        except Exception as err:
            resstock_ui = str(err)

        try:
            comstock_ui = mo.ui.tabs(
                {
                    com_buildings[x]: (
                        y[[f"{z}[{'%' if fraction_ui.value else 'MW'}]" for z in ["baseload", "heating", "cooling"]]]
                        .resample("1d")
                        .mean()
                        .plot.area(
                            figsize=(15, 10),
                            color=["g", "r", "b"],
                            grid=True,
                            ylabel="Load [MW]",
                            xlabel=f"Date/Time ({timezone.name})"
                        )
                        if graph_ui.value
                        else y.round(1)
                    )
                    for x, y in comstock.items() if isinstance(y,pd.DataFrame)
                },
                lazy=True,
            )
        except Exception as err:
            comstock_ui = str(err)

        _daily = weather["temperature[degC]"].resample("1d")
        _data = pd.DataFrame(
            {"Min": _daily.min(), "Mean": _daily.mean(), "Max": _daily.max()}
        )
        weather_ui = (
            _data.plot(figsize=(15, 10), ylabel="Daily temperature [$^\circ$C]", grid=True, xlabel=f"Date/Time ({timezone.name})")
            if graph_ui.value
            else weather
        )
    return comstock_ui, resstock_ui, weather_ui


@app.cell
def _(county, load_models, mo):
    # NERC model
    import matplotlib.pyplot as plt
    import numpy as np
    _model = load_models.NERCModel(county)
    _weather = _model.weather.data
    _loads = _model.loads.data
    _data = _weather.join(_loads)
    _model.fit()
    _x = np.arange(_data["temperature[degC]"].min(),_data["temperature[degC]"].max())
    _y = _model.predict(_x)
    _model.results
    plt.figure(figsize=(10,5))
    plt.plot(_data["temperature[degC]"],_data["total[MW]"],'.')
    plt.plot(_x,_y,"-k")
    plt.grid()
    nerc_model_plot = plt.gca()
    _summaries = [f"<tr><th>{x}</th><td>{round(y,1)}</td></tr>" for x,y in _model.results.items() if isinstance(y,float)]
    nerc_model_text = mo.md(f"<table><caption><u>Fit results</u></caption>{''.join(_summaries)}</table>")
    return nerc_model_plot, nerc_model_text


@app.cell
def _(graph_ui, nerc_model_plot, nerc_model_text):
    nerc_model = nerc_model_plot if graph_ui.value else nerc_model_text
    return (nerc_model,)


@app.cell
def _(mo, nerc_model):
    # Models UI
    models_ui = mo.ui.tabs(
        {
            "NERC": nerc_model,
        }, lazy=True
    )
    return (models_ui,)


@app.cell
def _(comstock, graph_ui, mo, pd, resstock, timezone):
    # Totals UI
    try:
        totals = pd.concat(
            [
                sum(
                    [
                        sum([x[f"{z}[MW]"] for x in y.values() if isinstance(x,pd.DataFrame)])
                        for y in [resstock, comstock]
                    ]
                )
                for z in ["baseload", "heating", "cooling","total"]
            ],
            axis=1,
        )
        totals_ui = (
            totals.resample("1d")
            .mean()
            .plot.area(
                figsize=(15, 10),
                grid=True,
                color=["g", "r", "b"],
                xlabel=f"Date/Time ({timezone.name})",
                ylabel="Total load [MW]",
            )
            if graph_ui.value
            else totals.round(1)
        )
    except Exception as err:
        tutals_ui = mo.md(err)
    return (totals_ui,)


@app.cell
def _(mo, set_errors):
    def clear_errors(x):
        set_errors({})
    clear_ui = mo.ui.button(label="Clear",on_change=clear_errors)
    return (clear_ui,)


@app.cell
def _(clear_ui, get_errors, mo):
    # Error logs
    errors_ui = mo.vstack(
        [
            mo.hstack([mo.md("## Processing results"),clear_ui]),
            mo.accordion(
                {
                    f"{x.title()} ({len(y)} reports)": mo.md(
                        "\n".join([f"{n+1}. ERROR: {z}" for n, z in enumerate(y)])
                    )
                    for x, y in get_errors().items()
                }
            ) if get_errors() else mo.md("No reports"),
        ]
    )
    return (errors_ui,)


@app.cell
def _():
    import marimo as mo
    import os
    import sys
    import datetime as dt
    import pandas as pd
    import load_models
    from tzinfo import TIMEZONES, TZ
    TZINFO={
        "EST" : TZ("EST",-5,0),
        "CST" : TZ("CST",-6,0),
        "MST" : TZ("MST",-7,0),
        "PST" : TZ("PST",-8,0),
    }
    return TIMEZONES, TZINFO, load_models, mo, pd, sys


if __name__ == "__main__":
    app.run()
