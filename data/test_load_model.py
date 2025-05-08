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
def _(counties, mo):
    # select state
    _options = counties.usps.unique()
    state_ui = mo.ui.dropdown(
        label="State:", options=_options, value=list(_options)[0]
    )
    return (state_ui,)


@app.cell
def _(pd):
    counties = pd.read_csv("counties.csv",index_col="geocode")
    return (counties,)


@app.cell
def _(counties, mo, state_ui):
    _options = {
        y.county: x for x, y in counties.iterrows() if y.usps == state_ui.value
    }
    get_errors, set_errors = mo.state({})


    def add_error(group,message):
        errors = get_errors()
        if group not in errors:
            errors[group] = []
        errors[group].append(message)
        set_errors(errors)


    county_ui = mo.ui.dropdown(
        label="County:",
        options=_options,
        value=list(_options)[0],
        on_change=lambda x: set_errors({}),
    )
    return add_error, county_ui, get_errors


@app.cell
def _(mo):
    graph_ui = mo.ui.checkbox(label="Show graph",value=True)
    fraction_ui = mo.ui.checkbox(label="Show fractional load")
    return fraction_ui, graph_ui


@app.cell
def _(county_ui, fraction_ui, graph_ui, mo, state_ui):
    mo.hstack([state_ui,county_ui,graph_ui,fraction_ui],justify='start')
    return


@app.cell
def _(
    comstock_ui,
    errors_ui,
    mo,
    models_ui,
    resstock_ui,
    totals_ui,
    weather_ui,
):
    mo.ui.tabs({
        "Residential" : resstock_ui,
        "Commercial" : comstock_ui,
        "Weather" : weather_ui,
        "Totals" : totals_ui,
        "Models" : models_ui,
        "Logs": errors_ui,
        },
        lazy=True)
    return


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
    return fips, timezone


@app.cell
def _(dt, fips, os, pd, state_ui, timezone):
    # weather data
    _server = "https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2021/resstock_amy2018_release_1/weather/amy2018/{fips}.csv"
    _name = f"g{fips[:2]}0{fips[2:]}0-weather"
    _file = f"geodata/rawdata/{state_ui.value}/{_name}.csv"
    if not os.path.exists(_file):
        _url = _server.format(fips=f"G{fips[:2]}0{fips[2:]}0_2018")
        _data = pd.read_csv(_url,
            index_col = [0],
            usecols = [0,1,3,5],
            parse_dates = [0],
            dtype = float,
            low_memory = True,
            header=None,
            skiprows=1,
            )
        _data.columns = ["temperature[degC]","wind[m/s]","solar[W/m^2]"]
        _data.index = _data.index.tz_localize(timezone).tz_convert("UTC").tz_localize(None)-dt.timedelta(hours=1) # localize and change to leading timestamp
        _data.index.name = "timestamp"
        weather = _data.resample("1h").mean().round(1)
        weather.to_csv(_file,header=True,index=True)
    else:
        weather = pd.read_csv(_file,index_col=[0],parse_dates=[0],low_memory=True)

    return (weather,)


@app.cell
def _(add_error, dt, fips, mo, os, pd, state_ui, timezone):
    # residential building data
    _server = "https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2021/resstock_amy2018_release_1/timeseries_aggregates/by_county/state={usps}/{fips}.csv"
    res_buildings = {
        "single-family_detached" : "House",
        "single-family_attached" : "Townhouse",
        "multi-family_with_2_-_4_units" : "Small apartment/condo",
        "multi-family_with_5plus_units" : "Large apartment/condo",
        "mobile_home" : "Mobile home",
    }
    resstock = {}
    with mo.status.progress_bar(
        collection=res_buildings,
        title="Loading residential building data...",
        remove_on_exit=True,
    ) as _bar:
        for _building in res_buildings:
            _name = f"g{fips[:2]}0{fips[2:]}0-{_building}"
            _file = f"geodata/rawdata/{state_ui.value}/{_name}.csv"
            if os.path.exists(_file):
                _bar.update(subtitle=f"Reading {_name}...")
                _data = pd.read_csv(
                    _file, index_col=[0], parse_dates=[0], low_memory=True
                )
            else:
                _repo = _server.format(usps=state_ui.value, fips=_name)
                _bar.update(subtitle=f"Downloading {_name}...")

                try:
                    _data = pd.read_csv(
                        _repo,
                        index_col=["timestamp"],
                        usecols=[
                            "timestamp",
                            "in.geometry_building_type_recs",
                            "out.electricity.cooling.energy_consumption",
                            "out.electricity.heating.energy_consumption",
                            "out.electricity.heating_supplement.energy_consumption",
                            "out.electricity.total.energy_consumption",
                        ],
                        parse_dates=["timestamp"],
                        converters={
                            "out.electricity.cooling.energy_consumption": lambda x: float(
                                x
                            )
                            / 1000,
                            "out.electricity.heating.energy_consumption": lambda x: float(
                                x
                            )
                            / 1000,
                            "out.electricity.heating_supplement.energy_consumption": lambda x: float(
                                x
                            )
                            / 1000,
                            "out.electricity.total.energy_consumption": lambda x: float(
                                x
                            )
                            / 1000,
                        },
                        low_memory=True,
                    )
                    _data.columns = [
                        "building_type",
                        "cooling[MW]",
                        "heating[MW]",
                        "auxheat[MW]",
                        "total[MW]",
                    ]
                    _data["heating[MW]"] += _data["auxheat[MW]"]
                    _data.drop("auxheat[MW]", axis=1, inplace=True)
                    _data.index = (
                        _data.index.tz_localize(timezone)
                        .tz_convert("UTC")
                        .tz_localize(None)
                    )
                    _data.index = _data.index - dt.timedelta(
                        minutes=15
                    )  # change lagging timestamps to leading timestamp
                    _data = pd.DataFrame(_data.resample("1h").sum())
                    _data["building_type"] = _building
                    os.makedirs(f"geodata/rawdata/{state_ui.value}", exist_ok=True)
                    _data.round(6).to_csv(_file, header=True, index=True)
                except Exception as err:
                    add_error("residential",f"unable to download {_name} from {_repo} failed ({err})")
                    _data = err
            if isinstance(_data, pd.DataFrame):
                _data["baseload[MW]"] = (
                    _data["total[MW]"]
                    - _data["heating[MW]"]
                    - _data["cooling[MW]"]
                )
                _data["baseload[%]"] = _data["baseload[MW]"] / _data["total[MW]"] * 100
                _data["heating[%]"] = _data["heating[MW]"] / _data["total[MW]"] * 100
                _data["cooling[%]"] = _data["cooling[MW]"] / _data["total[MW]"] * 100
            resstock[_building] = (
                mo.md(str(_data))
                if isinstance(_data,Exception)
                else _data.drop(["total[MW]", "building_type"], axis=1)
            )
    return res_buildings, resstock


@app.cell
def _(add_error, dt, fips, mo, os, pd, state_ui):
    # commercial building data
    _server = "https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2021/comstock_amy2018_release_1/timeseries_aggregates/by_county/state={usps}/{fips}.csv"
    com_buildings = {
        "largeoffice": "Large office",
        "secondaryschool" : "Large school",
        "largehotel" : "Large hotel",
        "hospital" : "Hospital",
        "mediumoffice" : "Medium Office",
        "retailstripmall" : "Medium retail",
        "outpatient" : "Healthcare",
        "smalloffice" : "Small office",
        "retailstandalone" : "Small retail",
        "primaryschool" : "Small school",
        "smallhotel" : "Small hotel",
        "fullservicerestaurant" : "Restaurant",
        "quickservicerestaurant" : "Fast food",
        "warehouse" : "Warehouse",
    }
    comstock = {}
    with mo.status.progress_bar(
        collection=com_buildings,
        title="Loading commercial building data...",
        remove_on_exit=True,
    ) as _bar:
        for _building in com_buildings:
            _name = f"g{fips[:2]}0{fips[2:]}0-{_building}"
            _file = f"geodata/rawdata/{state_ui.value}/{_name}.csv"
            if os.path.exists(_file):
                _bar.update(subtitle=f"Reading {_name}...")
                _data = pd.read_csv(
                    _file, index_col=[0], parse_dates=[0], low_memory=True
                )
            else:
                _repo = _server.format(usps=state_ui.value, fips=_name)
                _bar.update(subtitle=f"Downloading {_name}...")

                try:
                    _data = pd.read_csv(
                        _repo,
                        index_col=["timestamp"],
                        usecols=[
                            "timestamp",
                            "in.geometry_building_type_recs",
                            "out.electricity.cooling.energy_consumption",
                            "out.electricity.heating.energy_consumption",
                            "out.electricity.heating_supplement.energy_consumption",
                            "out.electricity.total.energy_consumption",
                        ],
                        parse_dates=["timestamp"],
                        converters={
                            "out.electricity.cooling.energy_consumption": lambda x: float(
                                x
                            )
                            / 1000,
                            "out.electricity.heating.energy_consumption": lambda x: float(
                                x
                            )
                            / 1000,
                            "out.electricity.heating_supplement.energy_consumption": lambda x: float(
                                x
                            )
                            / 1000,
                            "out.electricity.total.energy_consumption": lambda x: float(
                                x
                            )
                            / 1000,
                        },
                        low_memory=True,
                    )
                    _data.columns = [
                        "building_type",
                        "cooling[MW]",
                        "heating[MW]",
                        "auxheat[MW]",
                        "total[MW]",
                    ]
                    _data["heating[MW]"] += _data["auxheat[MW]"]
                    _data.drop("auxheat[MW]", axis=1, inplace=True)
                    _data.index = (
                        _data.index.tz_localize("EST")
                        .tz_convert("UTC")
                        .tz_localize(None)
                    )
                    _data.index = _data.index - dt.timedelta(
                        minutes=15
                    )  # change lagging timestamps to leading timestamp
                    _data = pd.DataFrame(_data.resample("1h").sum())
                    _data["building_type"] = _building
                    os.makedirs(f"geodata/rawdata/{state_ui.value}", exist_ok=True)
                    _data.round(6).to_csv(_file, header=True, index=True)
                except Exception as err:
                    add_error("commercial",f"unable to download {_name} from {_repo} failed ({err})")
                    _data = err
            if isinstance(_data, pd.DataFrame):
                _data["baseload[MW]"] = (
                    _data["total[MW]"]
                    - _data["heating[MW]"]
                    - _data["cooling[MW]"]
                )
                _data["baseload[%]"] = _data["baseload[MW]"] / _data["total[MW]"] * 100
                _data["heating[%]"] = _data["heating[MW]"] / _data["total[MW]"] * 100
                _data["cooling[%]"] = _data["cooling[MW]"] / _data["total[MW]"] * 100
            comstock[_building] = (
                mo.md(str(_data))
                if isinstance(_data,Exception)
                else _data.drop(["total[MW]", "building_type"], axis=1)
            )
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
def _(graph_ui, totals, weather):
    _data = weather.join(totals)
    _data["total[MW]"] = _data["baseload[MW]"] + _data["heating[MW]"] + _data["cooling[MW]"]
    nerc_model = _data.plot.scatter("temperature[degC]","total[MW]",1,figsize=(15,10),xlabel="Temperature [$^\circ$C]",ylabel="Total load [MW]",grid=True) if graph_ui.value else _data.round(1)
    return (nerc_model,)


@app.cell
def _(mo, nerc_model):
    models_ui = mo.ui.tabs(
        {"NERC": nerc_model, "Quantile": "TODO", "Linear": "TODO"}, lazy=True
    )
    return (models_ui,)


@app.cell
def _(comstock, graph_ui, mo, pd, resstock, timezone):
    try:
        totals = pd.concat(
            [
                sum(
                    [
                        sum([x[f"{z}[MW]"] for x in y.values() if isinstance(x,pd.DataFrame)])
                        for y in [resstock, comstock]
                    ]
                )
                for z in ["baseload", "heating", "cooling"]
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
    return totals, totals_ui


@app.cell
def _(get_errors, mo):
    errors_ui = mo.vstack(
        [
            mo.md("## Processing results"),
            mo.accordion(
                {
                    f"{x.title()} ({len(y)} reports)": mo.md(
                        "\n".join([f"{n+1}. ERROR: {z}" for n, z in enumerate(y)])
                    )
                    for x, y in get_errors().items()
                }
            ),
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
    from tzinfo import TIMEZONES, TZ
    TZINFO={
        "EST" : TZ("EST",-5,0),
        "CST" : TZ("CST",-6,0),
        "MST" : TZ("MST",-7,0),
        "PST" : TZ("PST",-8,0),
    }
    return TIMEZONES, TZINFO, dt, mo, os, pd


if __name__ == "__main__":
    app.run()
