import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    This notebook rescales the county level loads from 2018 based on the state-level energy consumption growth as reported by EIA (source?).

    The EIA annual energy use by state and year is available for all sectors. However, the NREL load model only applies to the RES and COM sector. Consequently, the EIA energy data for the other sectors (most notably IND) will be assumed to arise from constant power over each year. Only the RES and COM sectors will be scaled based on the time-varying NREL load model.
    """
    )
    return


@app.cell
def _(eia_ui, mo, projection_ui):
    mo.accordion({
        "EIA Energy Use Data":eia_ui,
        "Load Model Projection":projection_ui,
    }, multiple=True)
    return


@app.cell
def _(pd):
    new = pd.read_csv("wecc240_loads.csv",index_col=[0])
    new.index.name="datetime"
    return (new,)


@app.cell
def _(new, node_state, pd):
    _data = pd.DataFrame(new.stack().reset_index())
    _data.columns = ["datetime", "geocode", "model"]
    _data["datetime"] = pd.DatetimeIndex(_data["datetime"])
    _data["state"] = [node_state[x] for x in _data["geocode"]]
    _data["year"] = [x.year for x in _data["datetime"]]
    _data.set_index(["year", "state"], inplace=True)
    model_data = pd.DataFrame(
        (_data["model"].groupby(["year", "state"]).sum() / 1e6).round(1) # kWh->GWh
    )
    return (model_data,)


@app.cell
def _(mo, model_data, states):
    model_ui = mo.ui.table(model_data,page_size=len(states),selection=None)
    return (model_ui,)


@app.cell
def _(eia_data, mo, model_data, pd, states):
    # Compute the projection of building and industry loads
    _buildings = model_data.join(
        (
            eia_data.loc[states, :, ["COM", "RES"]]
            .groupby(["year", "state"])
            .sum()
            / 1000 # MWh->GWh
        ).round(1)
    )
    _buildings.columns = ["model", "eia"]
    _buildings["scalar"] = (_buildings.eia / _buildings.model).round(3)
    _buildings
    _industry = eia_data.loc[states,:,["IND","TRA","OTH"]].groupby(["year","state"]).sum() / 1000
    _industry.columns=["constant"]
    _industry["constant"] = (_industry["constant"] / _buildings["scalar"]).round(1)
    projection = pd.DataFrame(_buildings.join(_industry))[["scalar","constant"]]
    scaling_ui = mo.ui.table(projection,page_size=len(states),selection=None)
    return (scaling_ui,)


@app.cell
def _(mo, model_ui, scaling_ui):
    projection_ui = mo.ui.tabs({
        "Building model" : model_ui,
        "Projection model": scaling_ui,
    })
    return (projection_ui,)


@app.cell
def _(pd):
    counties = pd.read_csv("../data/counties.csv",index_col=["geocode"])
    # counties
    return (counties,)


@app.cell
def _(counties, new, utils):
    node_state = {x:counties.loc[utils.nearest(x,counties.index)].usps for x in new.columns}
    states = [x for x in counties.usps.unique() if x in node_state.values()]
    # states,node_state
    return node_state, states


@app.cell(hide_code=True)
def _(mo, pd, states):
    # Load EIA energy use data by state and year
    eia_data = pd.read_csv(
        "../data/EIA/eia_all_sectors.csv", index_col=["state", "year", "sector"]
    )
    years = eia_data.loc[states, :, :].reset_index().year.unique().tolist()
    state_ui = mo.ui.dropdown(label="State:",options=states,value=states[0])
    year_ui = mo.ui.dropdown(label="Year:",options=years,value=years[0])
    return eia_data, state_ui, year_ui


@app.cell(hide_code=True)
def _(eia_data, mo, state_ui, states, year_ui):
    # Show EIA energy use data for selected state and year
    _eiadata = (
        (eia_data.loc[states, :, :] / 1000)
        .groupby(["year", "sector"])
        .sum()
        .unstack()
    )
    _eiadata.columns = [x[1] for x in _eiadata.columns.tolist()]
    _statedata = eia_data.loc[state_ui.value, year_ui.value, :] / 1000
    _statedata.loc["TOTAL"] = _statedata.sum()
    _statedata.columns=["Energy [TWh/y]"]

    eia_ui = mo.vstack(
        [
            mo.hstack([state_ui, year_ui], justify="start"),
            mo.ui.tabs(
                {
                    "WECC Totals": _eiadata.plot(
                        kind="area",
                        ylabel="Energy [TWh/y]",
                        xlabel="Year",
                        grid=True,
                        title=f"WECC",
                        legend=eia_data.index.get_level_values(2)
                        .unique()
                        .tolist(),
                    ),
                    f"WECC {year_ui.value} Totals": eia_data.loc[
                        states, year_ui.value, :
                    ]
                    .groupby("sector")
                    .sum()
                    .plot(
                        kind="pie",
                        subplots=True,
                        title=f"WECC {year_ui.value}",
                        ylabel="",
                        xlabel="",
                    )[0],
                    f"{state_ui.value} {year_ui.value} Plot": eia_data.loc[
                        state_ui.value, year_ui.value, :
                    ].plot(
                        kind="pie",
                        subplots=True,
                        title=f"{state_ui.value} {year_ui.value}",
                        ylabel="",
                        xlabel="",
                    )[0],
                    f"{state_ui.value} {year_ui.value} Data": _statedata.round(1),
                }
            ),
        ]
    )
    return (eia_ui,)


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import sys
    sys.path.append("../data")
    import utils
    return mo, pd, utils


if __name__ == "__main__":
    app.run()
