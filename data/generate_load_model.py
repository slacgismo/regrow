import marimo

__generated_with = "0.13.6"
app = marimo.App(width="full", app_title="REGROW Load Model Generation")


@app.cell
def _(mo):
    mo.md(r"""This notebook generate the node-level load models and load data from the county-level data.""")
    return


@app.cell
def _(mo):
    mo.md(r"""# County-Level Load Model""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    County-level loads are collected from NREL's ResStock and ComStock databases for the year 2018. In addition, weather data for each county is also collected.  The load data and weather are used to construct a load model which estimates the static, temporal, and autoregressive components of the load behavior such that loads can be estimated for any combination of time and weather.


    **TODO: add final model fit and predict formulations here**

    *NOTE: what follows is not the test model currently implemented in the load_models module. This is the NERC 10-year load forecating model, which we will use to compare the results of the final model against.*

    The NREL electric load data comes in three separate components, baseload, heating, and cooling. The NERC load model is implemented a 192 3-part piece-wise linear fits for heating, baseload, and cooling components of the load. This yields a separate model for each hour of each daytype (weekdays and weekends/holidays) for each season (winter, spring, summer, fall). The heating and cooling loads are each fit in two parts such that the heating segment has a negative slope and the cooling segment has a positive slope. These fits are used to determine the heating and cooling balance temperatures, which serve as the knot values for the final 3-part fit on the total load. 

    The models are applied to each hour of the prediction window by identifying the hour, daytype, and season to select which model is used, and the temperature is identified from weather data, which is used to predict the load.
    """
    )
    return


@app.cell
def _(counties, lm, mo):
    # Load county-level load data from NREL

    lm.WARNING = False  # silence warnings about missing data from NREL

    _counties = {
        x: lm.County(x)
        for x in counties.data[counties.data.usps.isin(counties.regions["WECC"])].geocode
    }
    weather = {}
    loads = {}

    # collect county weather and load data
    with mo.status.progress_bar(
        title="Loading county data...", total=len(_counties), remove_on_exit=True
    ) as _bar:
        for _geocode, _county in _counties.items():
            _refresh = "auto" # "auto", "never", or "always"
            _bar.update(subtitle=f"Processing {_county}...")

            weather[_geocode] = lm.Weather(_county)
            try:
                loads[_geocode] = lm.Loads(_county,download=_refresh)
            except RuntimeError as err:
                print(f"WARNING: no data for {_county}")
    return loads, weather


@app.cell
def _(counties, mo):
    _options = {f"{y.county} {y.usps}":x for x,y in counties.data.set_index(["region","geocode"]).loc["WECC"].iterrows()}
    county_ui = mo.ui.dropdown(options=_options,value=list(_options)[0])
    return (county_ui,)


@app.cell
def _(county_ui, lm, loads, mo, np, weather):
    _figure = (
        weather[county_ui.value]
        .data.join(loads[county_ui.value].data)
        .plot.scatter(
            "temperature[degC]",
            "total[MW]",
            1,
            figsize=(10, 7),
            grid=True,
            xlabel="Temperature [$^\\circ$C]",
            ylabel="Building Load [MW]",
            label="NREL data",
        )
    )

    _model = lm.NERCModel(weather[county_ui.value],loads[county_ui.value])
    _model.fit(cutoff=1.0)

    _temps = np.arange(-10,40)
    _figure.plot(_temps,_model.predict(_temps),"-k",label="NERC model")
    _figure.legend()

    mo.vstack(
        [
            _figure,
            mo.md(
                f"**Figure 1: Building load versus temperature for {county_ui} in 2018**"
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    County-level loads must be calibrated to state EIA energy use for the target years. The scaling ratio for each state and each year is the total energy use reported by EIA divided by total energy use on the load model. However the load model only includes residential and commercial energy use, which accounts for only a portion of the total energy use. Therefore, the industrial, transportation, and other energy use must be excluded from the scaling.

    $\qquad F_{ito}[x,y] = \frac{E_{ito}[x,y]}{E_{total}[s,y]}$ and $F_{rc}[x,y] = \frac{E_{rc}[x,y]}{E_{total}[x,y]}$ for $x \in \{states\}$ and $y \in \{years\}$

    where

    * $E_{rc}[x,y]$ is the state $x$ EIA building energy use for the year $y$,
    * $E_{ito}[x,y]$ is the state $x$ industrial, transportation, and other energy use for the year $y$,
    * $E_{total}[x,y]$ is the state $x$ total energy use for the year $y$,
    * $F_{rc}[x,y]$ is the fraction of energy use that goes to buildings in the state $x$ and the year $y$,
    * $F_{ito}[x,y]$ is the fraction of energy use that does not go to buildings in the state $x$ and the year $y$.
    """
    )
    return


@app.cell
def _(eia, states):
    # load EIA annual energy use data
    ref_year = 2018
    _eiadata = {x:eia.StateEnergyProfile(x).data for x in states}
    energy = {
        y: {
            "eia_buildings[MWh]": sum(
                [
                    _eiadata[y]["sales[MWh]"][x][ref_year]
                    for x in ["Residential", "Commercial"]
                    if ref_year in _eiadata[y]["sales[MWh]"][x]
                ]
            ),
            "eia_others[MWh]": sum(
                [
                    _eiadata[y]["sales[MWh]"][x][ref_year]
                    for x in ["Industrial", "Transportation", "Other"]
                    if ref_year in _eiadata[y]["sales[MWh]"][x]
                ]
            ),
            "eia_total[MWh]": _eiadata[y]["sales[MWh]"]["Total"][ref_year],
        }
        for y in states
    }
    return (energy,)


@app.cell
def _(counties, lm):
    # group counties by state for state-level aggregation and scaling
    county = {}
    states = {
        x: {
            y: lm.County(y)
            for y in counties.data.set_index(["region", "usps"])
            .sort_index()
            .loc[("WECC", x), "geocode"]
            .values
        }
        for x in counties.regions["WECC"]
    }

    return county, states


@app.cell
def _(counties, county, energy, lm, loads, pd):
    # calculate state-level energy use to scale model data to EIA levels
    _states = counties.data[
        counties.data.usps.isin(counties.regions["WECC"])
    ].set_index("usps")
    _states = {
        x: {y: lm.County(y) for y in _states.loc[x]["geocode"].values}
        for x in _states.index.unique()
    }
    for _state, _counties in _states.items():
        energy[_state].update({"model_buildings[MWh]": 0.0})
        for _geocode, _county in _counties.items():
            try:
                county[_geocode] = _county
                energy[_state]["model_buildings[MWh]"] += float(
                    loads[_geocode].data["total[MW]"].sum()
                )
                energy[_state]["model_scale[pu]"] = (
                    energy[_state]["eia_buildings[MWh]"]
                    / energy[_state]["model_buildings[MWh]"]
                )
                energy[_state]["F_buildings[pu]"] = (
                    energy[_state]["eia_buildings[MWh]"] / energy[_state]["eia_total[MWh]"]
                )
                energy[_state]["F_others[pu]"] = (
                    energy[_state]["eia_others[MWh]"] / energy[_state]["eia_total[MWh]"]
                )
                energy[_state]["model_total[MWh]"] = (
                    energy[_state]["model_buildings[MWh]"] * energy[_state]["model_scale[pu]"] / energy[_state]["F_buildings[pu]"]
                )
                energy[_state]["model_others[MWh]"] = energy[_state]["model_total[MWh]"] * energy[_state]["F_others[pu]"]
            except Exception as err:
                print(f"WARNING: {_county} data not found ({err})")
                pass
        # energy[_state]["model_buildings"] = round(
        #     energy[_state]["model_buildings"], 0
        # )
        # print(_state, energy[_state])
    scaling = {2018:pd.DataFrame(energy).transpose().round(3)}
    scaling[2018].to_csv("eia_scaling_2018.csv",header=True,index=True)
    scaling[2018]
    return


@app.cell
def _(pd):
    # load the nodes for which data must be aggregated
    nodes = list(pd.read_csv("geodata/temperature.csv",index_col="timestamp").columns)
    return (nodes,)


@app.cell
def _(counties, loads, nodes, pd, wecc):

    # project loads to target years
    years = list(range(2018,2023))
    for year in years:

        # get weather for years

        # predict loads

        pass

    # aggregate to nodes
    data = {}
    for node in nodes:
        data[node] = []
        for _county in counties.find_counties(node,counties,list(wecc),nodes):
            print(_county,"-->",node)
            data[node].append(loads)
        data[node] = pd.concat(data[node])
    return


@app.cell
def _():
    import os
    import sys
    import marimo as mo
    import numpy as np
    import pandas as pd
    import load_models as lm
    import counties
    import eia
    return counties, eia, lm, mo, np, pd


if __name__ == "__main__":
    app.run()
