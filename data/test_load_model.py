import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    This notebook tests different load models relative to a simple "bathtub" load model.

    - [x] Read load and weather data
    - [x] Hold out 7 days each month
    - [x] Run the original county-level project method
    - [ ] Benchmark the bathtub model for the selected county (RMSE and % error tests) on hold out data
    - [ ] Implement scaling for load growth, upgrades, and electrification.
    - [ ] Implement experimental models
    - [ ] Zoom controls on plots to see hourly behavior (change plot module?)
    """
    )
    return


@app.cell
def _(county_ui, fraction_ui, graph_ui, holdout_ui, mo, state_ui, wecc_ui):
    mo.hstack([wecc_ui,state_ui,county_ui,graph_ui,fraction_ui,holdout_ui],justify='start')
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
    # error log handling
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
    # county selection
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
    # county/graphing options
    wecc_ui = mo.ui.checkbox(label="WECC",value=False)
    graph_ui = mo.ui.checkbox(label="Show graph",value=False)
    fraction_ui = mo.ui.checkbox(label="Show fractional load")
    holdout_ui = mo.ui.checkbox(label="No holdout test")
    return fraction_ui, graph_ui, holdout_ui, wecc_ui


@app.cell
def _(county_ui, load_models, mo):
    # county selection check
    mo.stop(county_ui.value==None,mo.md("**HINT**: Select a county"))
    county = load_models.County(county_ui.value)
    weather = load_models.Weather(county).data
    return county, weather


@app.cell
def _(add_error, county, load_models, mo):
    # data loads
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
    county,
    fraction_ui,
    graph_ui,
    mo,
    pd,
    res_buildings,
    resstock,
    weather,
):
    # Generate tab contents
    _units = '%' if fraction_ui.value else 'MW'
    with mo.status.spinner(
        title="Generating plots" if graph_ui.value else "Updating tables",
        remove_on_exit=True,
    ) as _spinner:
        try:
            resstock_ui = mo.ui.tabs(
                {
                    res_buildings[x]: (
                        y[[f"{z}[{_units}]" for z in ["baseload", "heating", "cooling"]]]
                        .resample("1d")
                        .mean()
                        .plot.area(
                            figsize=(15, 10),
                            color=["g", "r", "b"],
                            grid=True,
                            ylabel=f"Load [{_units}]",
                            xlabel=f"Date/Time ({county.timezone.name})",
                            ylim=[0,100] if fraction_ui.value else None,
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
                        y[[f"{z}[{_units}]" for z in ["baseload", "heating", "cooling"]]]
                        .resample("1d")
                        .mean()
                        .plot.area(
                            figsize=(15, 10),
                            color=["g", "r", "b"],
                            grid=True,
                            ylabel=f"Load [{_units}]",
                            xlabel=f"Date/Time ({county.timezone.name})",
                            ylim=[0,100] if fraction_ui.value else None,
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
            _data.plot(figsize=(15, 10), ylabel="Daily temperature [$^\circ$C]", grid=True, xlabel=f"Date/Time ({county.timezone.name})")
            if graph_ui.value
            else weather
        )
    return comstock_ui, resstock_ui, weather_ui


@app.cell
def _(holdout_ui):
    # hold out data (7 days out of 30 days)
    holdout = []
    if not holdout_ui.value:
        for month in range(12):
            holdout.extend(range(month*30,month*30+7))
    return (holdout,)


@app.cell
def _(county, holdout, load_models, mo, np, plt):
    # NERC model
    _model = load_models.NERCModel(county)
    _weather = _model.weather.data
    _loads = _model.loads.data
    _data = _weather.join(_loads)
    _model.holdout = holdout
    _model.fit()
    _x = np.arange(_data["temperature[degC]"].min(),_data["temperature[degC]"].max())
    _y = _model.predict(_x)
    _model.results
    plt.figure(figsize=(10,5))
    plt.plot(_data["temperature[degC]"],_data["total[MW]"],'.')
    plt.plot(_x,_y,"-k")
    plt.grid()
    nerc_model_plot = plt.gca()
    _summaries = [f"<tr><th>{x}</th><td>{y}</td></tr>" for x,y in _model.results.items() if isinstance(y,str)]
    nerc_model_text = mo.md(f"<table><caption><u>Fit results</u></caption>{''.join(_summaries)}</table>")
    return nerc_model_plot, nerc_model_text


@app.cell
def _(graph_ui, nerc_model_plot, nerc_model_text):
    # NERC model UI
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
def _(comstock, county, graph_ui, mo, pd, resstock):
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
                xlabel=f"Date/Time ({county.timezone.name})",
                ylabel="Total load [MW]",
            )
            if graph_ui.value
            else totals.round(1)
        )
    except Exception as err:
        totals_ui = mo.md(err)
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
    import matplotlib.pyplot as plt
    import numpy as np
    return load_models, mo, np, pd, plt, sys


if __name__ == "__main__":
    app.run()
