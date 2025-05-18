import marimo

__generated_with = "0.13.6"
app = marimo.App(width="full", app_title="REGROW Load Model Generation")


@app.cell
def _(mo):
    mo.md(r"""This notebook generate the node-level load models and load data from the county-level data.""")
    return


@app.cell
def _(mo):
    mo.md(r"""# Load Model""")
    return


@app.cell
def _(mo):
    mo.md(r"""## County-Level Load Model""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    County-level loads are collected from NREL's ResStock and ComStock databases for the year 2018. In addition, weather data for each county is also collected.  The load data and weather are used to construct a load model which estimates the static, temporal, and autoregressive components of the load behavior such that loads can be estimated for any combination of time and weather.


    **TODO: add final model fit and predict formulations here**

    *NOTE: what follows is not the test model currently implemented in the `load_models` module. This is the NERC 10-year load forecating model, which we will use to compare the results of the final model against.*

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
            _refresh = "never" # "auto", "never", or "always"
            _bar.update(subtitle=f"Processing {_county}...")

            weather[_geocode] = lm.Weather(_county)
            try:
                loads[_geocode] = lm.Loads(_county,download=_refresh)
            except RuntimeError as err:
                print(f"WARNING: no data for {_county}")
    return loads, weather


@app.cell
def _(mo):
    mo.md(r"""## Model Performance""")
    return


@app.cell
def _(mo):
    mo.md(r"""The load model is tested using a hold out of 1 week each month of the 2018.""")
    return


@app.cell
def _(counties, mo):
    _options = {f"{y.county} {y.usps}":x for x,y in counties.data.set_index(["region","geocode"]).loc["WECC"].iterrows()}
    county_ui = mo.ui.dropdown(options=_options,value=list(_options)[0])
    return (county_ui,)


@app.cell
def _(county_ui, lm, loads, mo, weather):
    with mo.status.spinner("Evaluating model performance") as _spinner:
        model = lm.NERCModel(weather[county_ui.value], loads[county_ui.value])
        model.holdout = [
            x for x in weather[county_ui.value].index if x // (24 * 7) % 4 == 0
        ]
        print(f"{len(model.holdout)/len(model.weather.data)*100:.1f}% data holdout")
        model.fit(cutoff=1.0)
        model.predict(list(model.weather["temperature[degC]"]))
    return (model,)


@app.cell
def _(county_ui, loads, mo, model, np, weather):
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
    _temps = np.arange(-10,40)
    _figure.plot(_temps,model.predict(_temps),"-k",label="NERC model")
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
def _(county_ui, mo, model):
    _table = f"""<table>
    {"".join([f"<tr><th>{x}</th><td>{y}</td></tr>" for x,y in model.results.items() if isinstance(y,str)])}
    </table>
    """

    mo.vstack(
        [
            mo.md(f"**Table 1: Load model performance for {county_ui}**"),
            mo.md(_table),
        ]
    )
    return


@app.cell
def _(model):
    model.results
    return


@app.cell
def _(mo):
    mo.md(r"""## EIA Energy Use Adjustment""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    County-level loads are calibrated to state EIA energy use for the target years. The scaling ratio for each state and each year is the total energy use reported by EIA divided by total energy use in the load model. However the load model only includes residential and commercial energy use, which accounts for only a portion of the total energy use. Therefore, the industrial, transportation, and other energy use must be excluded from the scaling and added to the total load for the county based on the fraction of building energy use relative to the total energy use.

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
def _(states):
    # load EIA annual energy use data
    from eia import StateEnergyProfile as _profile
    ref_year = 2018
    _eiadata = {x:_profile(x).data for x in states}
    energy = {}
    for _year in range(2018,2023):
        energy[_year] = {
            _state: {
                "eia_buildings[MWh]": sum(
                    [
                        _eiadata[_state]["sales[MWh]"][x][_year]
                        for x in ["Residential", "Commercial"]
                        if _year in _eiadata[_state]["sales[MWh]"][x]
                    ]
                ),
                "eia_others[MWh]": sum(
                    [
                        _eiadata[_state]["sales[MWh]"][x][_year]
                        for x in ["Industrial", "Transportation", "Other"]
                        if _year in _eiadata[_state]["sales[MWh]"][x]
                    ]
                ),
                "eia_total[MWh]": _eiadata[_state]["sales[MWh]"]["Total"][_year],
            }
            for _state in states
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
        energy[2018][_state].update({"model_buildings[MWh]": 0.0})
        for _geocode, _county in _counties.items():
            try:
                county[_geocode] = _county
                energy[2018][_state]["model_buildings[MWh]"] += float(
                    loads[_geocode].data["total[MW]"].sum()
                )
                energy[2018][_state]["model_scale[pu]"] = (
                    energy[2018][_state]["eia_buildings[MWh]"]
                    / energy[2018][_state]["model_buildings[MWh]"]
                )
                energy[2018][_state]["F_buildings[pu]"] = (
                    energy[2018][_state]["eia_buildings[MWh]"] / energy[2018][_state]["eia_total[MWh]"]
                )
                energy[2018][_state]["F_others[pu]"] = (
                    energy[2018][_state]["eia_others[MWh]"] / energy[2018][_state]["eia_total[MWh]"]
                )
                energy[2018][_state]["model_total[MWh]"] = (
                    energy[2018][_state]["model_buildings[MWh]"] * energy[2018][_state]["model_scale[pu]"] / energy[2018][_state]["F_buildings[pu]"]
                )
                energy[2018][_state]["model_others[MWh]"] = energy[2018][_state]["model_total[MWh]"] * energy[2018][_state]["F_others[pu]"]
            except Exception as err:
                print(f"WARNING: {_county} data not found ({err})")
                pass
        # energy[_state]["model_buildings"] = round(
        #     energy[_state]["model_buildings"], 0
        # )
        # print(_state, energy[_state])
    scaling = {2018:pd.DataFrame(energy[2018]).transpose().round(3)}
    scaling[2018].to_csv("eia_scaling_2018.csv",header=True,index=True)
    return (scaling,)


@app.cell
def _(mo, scaling):
    from states import state_codes

    state_names = {x[1]: x[0] for x in state_codes}
    _table = scaling[2018][
        ["model_scale[pu]", "F_buildings[pu]", "F_others[pu]"]
    ].astype(str)
    for _column in _table.columns:
        if _column == "model_scale[pu]":
            _table.loc[_table.index, _column] = [
                f"{(float(x)-1)*100:+.0f}%" for x in _table[_column].values
            ]
        else:
            _table.loc[_table.index, _column] = [
                f"{float(x)*100:.0f}%" for x in _table[_column].values
            ]
    _table.columns = [
        "Energy Adjustment",
        "Buildings (See Note 1)",
        "Others (See Note 2)",
    ]
    _table.index = [state_names[x] for x in _table.index]
    _rows = [
        f"""<tr><td>{x}</td><td>{"</td><td>".join(y.values)}</td></tr>"""
        for x, y in _table.iterrows()
    ]
    mo.vstack(
        [
            mo.md("**Table 2: 2018 Load Model EIA Adjustments**"),
            mo.md(f"""<table>
    <th>State</th><th>{"</th><th>".join(_table.columns)}</th>
    {"".join(_rows)}
    </table>
    """),
            mo.md("""Notes:
        
    1. Buildings include all residential and commercial building types

    2. Others include industrial and transportation sectors""")
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""# Load Projection""")
    return


@app.cell
def _(mo):
    mo.md(r"""The load model obtained for 2018 is used to project loads given the observed weather for each study year in each county.""")
    return


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
def _(mo):
    mo.md(r"""## Node Aggregation""")
    return


@app.cell
def _(loads):
    # load the nodes for which data must be aggregated
    import json
    import utils

    wecc240 = json.load(open("../model/wecc240.json", "r"))
    objects = wecc240["objects"]
    busses = {
        x: utils.geohash(float(y["latitude"]), float(y["longitude"]))
        for x, y in objects.items()
        if "_N_" in x # this should be a class test, but that doesn't seem to always work right (?!?)
    }
    nodes = set(busses.values())
    load_map = {
        x: utils.nearest(x, nodes) for x in loads.keys()
    }
    load_busses = {
        y["parent"]: {
            "node": busses[y["parent"]],
            "model": round(abs(complex(y["S"].split()[0])), 1),
            # "load": sum([loads[x].data["total[MW]"].max().round(1) for x,z in load_map.items() if z==busses[y["parent"]]]),
        }
        for x, y in objects.items()
        if y["class"] == "load"
    }
    load_nodes = {busses[x] for x in load_busses}
    other_nodes = [x for x in nodes if x not in load_nodes]
    other_loads = [x for x in busses.values() if x not in set(load_map.values())]
    return (
        busses,
        load_busses,
        load_map,
        load_nodes,
        nodes,
        objects,
        other_loads,
        other_nodes,
        utils,
    )


@app.cell
def _(loads):
    loads["9w61k3"].data["total[MW]"].max()
    return


@app.cell
def _(
    busses,
    load_busses,
    load_map,
    load_nodes,
    mo,
    nodes,
    objects,
    other_loads,
    other_nodes,
):
    mo.md(f"""The WECC 240 model defines a total of {len(objects)} objects. Of these {len(busses)} are network busses located at {len(nodes)} geographic nodes, {len(load_busses)} of which are defined in the mode as load nodes. County load data is aggregated to {len(load_nodes)} load nodes according to their proximity to these nodes. 

    In cases where more than 1 bus is allocated to a node, the aggregate county load is allocated pro-rata the load originally given in the original WECC 240 model. Consequently only {len(set(load_map.values()))} load nodes have county loads allocated to them. 

    The remaining load nodes are presumed to be static industrial, transportation, or other loads that are not included in the county-level loads obtained from the NREL building data. These {len(other_loads)} static loads are aggregated and deducted from the county aggregation of the {len(other_nodes)} geographic nodes nearest them.
    """)
    return


@app.cell
def _(loads, nodes, pd, utils):
    node_data = pd.DataFrame([0.0]*len(nodes),list(nodes)).sort_index()
    node_data.index.name = "node"
    node_data.columns = ["total"]
    for _load,_data in loads.items():
        _node = utils.nearest(_load,node_data.index)
        node_data.loc[_node,"total"] += _data.data["total[MW]"].max().round(1)
    return (node_data,)


@app.cell
def _(node_data):
    node_data.sum()
    return


@app.cell
def _(load_busses, pd):
    load_data = pd.DataFrame(load_busses.values(),[int(x.split("_")[-1]) for x in load_busses.keys()])
    load_data.index.name = "bus"
    _load = load_data.groupby("node").sum()
    _load.columns = ["load"]
    _counts = load_data.groupby("node").count()
    _counts.columns = ["count"]
    load_data = load_data.reset_index().set_index("node").join(_load).reset_index().set_index("bus")
    load_data = load_data.reset_index().set_index("node").join(_counts).reset_index().set_index("bus")
    load_data["allocation"] = load_data["model"] / load_data["load"]
    load_data.sort_index(inplace=True)
    load_data
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
    return counties, lm, mo, np, pd


if __name__ == "__main__":
    app.run()
