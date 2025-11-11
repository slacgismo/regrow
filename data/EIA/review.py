import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""This notebook is used to review the EIA Form 861 load data and 930 demand data used to generate the load growth relative to 2018.""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # EIA Form 861 Data

    Annual electric power industry reports detailed data in Form 861, as shown in Table 1.

    Source: https://www.eia.gov/electricity/data/eia861/
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Notes:
    1. The 2019 demand data is anomalous insofar as it appears to be too small by a factor of 10. 
    2. The Washington state transportation data from 2019 onward appears to be too large by a factor of 10.
    """
    )
    return


@app.cell
def _(mo):
    fix_2019_ui = mo.ui.checkbox(label="Remove 2019 data:",value=True)
    fix_WATRA_ui = mo.ui.checkbox(label="Remove Washington state transportation data:",value=True)
    mo.hstack([fix_2019_ui,fix_WATRA_ui],justify="start")
    return fix_2019_ui, fix_WATRA_ui


@app.cell
def _(fix_2019_ui, fix_WATRA_ui, mo, pd):
    data = (pd.read_csv("eia_f861.csv", index_col=[1, 0],converters={"year":str}) / 1000000).round(4).sort_index()
    data.columns = [x[0:3].upper() for x in data.columns]
    if fix_2019_ui.value:
        for _x in data.index.get_level_values(0).unique():
            data.loc[_x,"2019"] = 0.5*data.loc[_x,"2018"] + 0.5* data.loc[_x,"2020"] #*= 10
    if fix_WATRA_ui.value:
        for _x in data.index.get_level_values(1).unique()[1:]:
            data.loc["WA",_x].TRA = float('NAN')#/= 10
    mo.vstack([
        mo.md("**Table 1: Form 861 Annual Energy Use (GWh/y)**"),
        mo.ui.table(data,page_size=len(data.index.get_level_values(1).unique()),selection=None)
        ])
    return (data,)


@app.cell
def _(mo):
    mo.md(r"""The load growth is computed by divide the energy use for each year by the energy use in 2018, as shown in Figure 1.""")
    return


@app.cell
def _(data, mo):
    _options = data.index.get_level_values(0).unique()
    state_ui = mo.ui.radio(options=_options, value=_options[0],inline=True)
    state_ui
    return (state_ui,)


@app.cell
def _(data, mo, state_ui):
    growth_by_year = data.loc[state_ui.value, :] / data.loc[state_ui.value, "2018"]
    _result = (growth_by_year * 100 - 100).plot(
        figsize=(15, 6),
        grid=True,
        xlabel="Year",
        ylabel="Load growth (% w.r.t 2018)",
        legend="outside",
    )
    mo.vstack(
        [
            _result,
            mo.md(
                f"**Figure 1: {state_ui.value} annual load growth relative for 2018.**"
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""The state-level load growth by end-use sector is shown in Figure 2 for the following year.""")
    return


@app.cell
def _(data, mo):
    _options = data.index.get_level_values(1).unique()
    year_ui = mo.ui.radio(label="Year:",options=[x for x in _options],value="2020",inline=True)
    year_ui
    return (year_ui,)


@app.cell
def _(data, mo, year_ui):
    growth_by_state = (
        data.reset_index().set_index(["year", "state"]).loc[year_ui.value, :]
        / data.reset_index().set_index(["year", "state"]).loc["2018", :]
    )
    _result = (growth_by_state * 100 - 100).plot(
        figsize=(15, 7),
        kind="bar",
        grid=True,
        xlabel="State",
        ylabel=f"Load growth (% w.r.t 2018)",
    )
    mo.vstack(
        [
            _result,
            mo.md(
                "**Figure 2: load growth by state for {year_ui.value} relative to 2018.**"
            ),
        ]
    )
    return (growth_by_state,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Results

    The fractional load growth w.r.t 2018 are shown in Table 2.
    """
    )
    return


@app.cell
def _(growth_by_state, mo):
    mo.vstack([
        mo.md("**Table 2: Fractional load growth by state for each end-use sector w.r.t. to 2018.**"),
        growth_by_state.round(3),
        ])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # EIA Form 930 Data

    Utility demand data by balancing authority and utility for IOUs is available from EIA Form 930, as shown in Figure 3.

    Source: https://s3.us-west-2.amazonaws.com/pudl.catalyst.coop/nightly/core_eia930__hourly_subregion_demand.parquet. 

    Documentation: https://catalystcoop-pudl.readthedocs.io/en/stable/data_sources/eia930.html
    """
    )
    return


@app.cell
def _(pd):
    demand = pd.read_csv("eia930.csv.gz",index_col=[1,2,0],parse_dates=[0]).sort_index()
    return (demand,)


@app.cell
def _(demand, mo):
    _options=demand.index.get_level_values(0).unique()
    region_ui=mo.ui.radio(label="Balancing authority:",options=_options,value=_options[0],inline=True)
    region_ui
    return (region_ui,)


@app.cell
def _(demand, mo, region_ui):
    _options=demand.loc[region_ui.value,:].index.get_level_values(0).unique()
    utility_ui=mo.ui.radio(label="Utility:",options=_options,value=_options[0],inline=True)
    utility_ui
    return (utility_ui,)


@app.cell
def _(demand, mo, region_ui, utility_ui):
    mo.vstack([
        (demand.loc[region_ui.value,utility_ui.value]/1e3).plot(grid=True,figsize=(15,7),legend=None),
        mo.md(f"**Figure 3: EIA Form 930 data for {utility_ui.value}.**")
        ])
    return


@app.cell
def _(data, demand, dt):
    form930 = (
        demand.loc[
            :,
            :,
            dt.datetime(2019, 1, 1, 0, 0, 0) : dt.datetime(
                2022, 12, 31, 23, 59, 59
            ),
        ]
        .groupby(["datetime_utc"])
        .sum()
    )
    form930["year"] = form930.index.year
    form930 = form930.groupby("year").sum() / 8766
    iou_fraction = form930.demand_reported_mwh.mean()/1000/(data.loc["CA"].TOT/8766*1000).mean()*100
    return form930, iou_fraction


@app.cell
def _(iou_fraction, mo):
    mo.md(rf"""The EIA Form 930 data provides the demand for IOUs in California, which represent {iou_fraction:.1f}% of the total energy demand in California reported by Form 861 data. The different growth rates for Form 861 data for the entire state and Form 960 data are shown in Figure 4.""")
    return


@app.cell
def _(data, form930, mo, np):
    _fit = np.polyfit(form930.index - 2018, form930.demand_reported_mwh, 1)[1]
    _data = form930.copy()
    _data.loc[2018] = _fit
    _data.sort_index(inplace=True)
    _data["Form 930"] = _data.demand_reported_mwh / _fit
    _data["Form 861"] = (data.loc["CA", :] / data.loc["CA", "2018"]).TOT.tolist()
    mo.vstack([
        (_data[["Form 861", "Form 930"]] * 100 - 100).sort_index().plot(
            figsize=(15,6),
            grid=True,
            ylabel="Load growth (% w.r.t 2018)",
            xticks=range(2018, 2023),
        ),
        mo.md("**Figure 4: California IOU and state-wide load growth as reported by EIA.**")
        ])
    return


@app.cell
def _():
    import os
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import datetime as dt
    return dt, mo, np, pd


if __name__ == "__main__":
    app.run()
