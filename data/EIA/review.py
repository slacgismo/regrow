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

    Annual electric power industry reports detailed data files.

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
    fix_2019_ui = mo.ui.checkbox(label="Fix 2019 data:",value=True)
    fix_WATRA_ui = mo.ui.checkbox(label="Fix Washington state transportation data:",value=True)
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
    mo.ui.table(data,page_size=len(data.index.get_level_values(1).unique()),selection=None)
    return (data,)


@app.cell
def _(data, mo):
    _options = data.index.get_level_values(0).unique()
    state_ui = mo.ui.radio(options=_options, value=_options[0],inline=True)
    state_ui
    return (state_ui,)


@app.cell
def _(data, state_ui):
    # print(data)
    growth_by_year = (
        data.loc[state_ui.value,:]
        / data.loc[state_ui.value,"2018"]
    )
    # print(growth)
    growth_by_year.plot(
        title=f"EIA Form 861 Data for {state_ui.value}",
        grid=True,
        xlabel="Year",
        ylabel="Load growth (%/y)",
        legend="outside",
    )
    return


@app.cell
def _(data, mo):
    _options = data.index.get_level_values(1).unique()
    year_ui = mo.ui.radio(label="Year:",options=[x for x in _options],value="2020",inline=True)
    year_ui
    return (year_ui,)


@app.cell
def _(data, year_ui):
    growth_by_state = (
        data.reset_index().set_index(["year", "state"]).loc[year_ui.value, :]
        / data.reset_index().set_index(["year", "state"]).loc["2018", :]
    )
    growth_by_state.plot(figsize=(15,7),kind="bar", grid=True,title=f"Load growth from 2018 to {year_ui.value}")
    return (growth_by_state,)


@app.cell
def _(mo):
    mo.md(r"""## Results""")
    return


@app.cell
def _(growth_by_state):
    growth_by_state.round(3)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # EIA Form 930 Data

    Utility demand data by balancing authority and utility. Source: https://s3.us-west-2.amazonaws.com/pudl.catalyst.coop/nightly/core_eia930__hourly_subregion_demand.parquet. Documentation: https://catalystcoop-pudl.readthedocs.io/en/stable/data_sources/eia930.html
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
def _(demand, region_ui, utility_ui):
    (demand.loc[region_ui.value,utility_ui.value]/1e3).plot(grid=True,figsize=(15,7),legend=None)
    return


@app.cell
def _(demand, dt, np, plt, region_ui, utility_ui):
    _data = (demand.loc[region_ui.value,utility_ui.value,:dt.datetime(2024,12,31,23,59,59)]/1e3)
    # print(_data)
    # _data.plot(figsize=(10,5),grid=True,xlabel="Date",ylabel="GW",legend=False)
    _data["year"] = _data.index.get_level_values(2).year
    _groupyear = _data.groupby("year")
    _mean = _groupyear.mean()
    _fit0 = np.polyfit(_mean.index-2018,_mean.values,0).flatten()
    _fit1 = np.polyfit(_mean.index-2018,_mean.values,1).flatten()
    print(f"{_fit0=},{_fit1=}")
    plt.figure()
    plt.plot(_mean,label="Annual mean (GW)")
    plt.plot(range(2018,2025),np.polyval(_fit0,np.arange(2018,2025)-2018),label="Mean value (GW)")
    plt.plot(range(2018,2025),np.polyval(_fit1,np.arange(2018,2025)-2018),label=f"Growth relative to 2018 ({_fit1[1]:.1f}GW{_fit1[0]*100:+.1f}%/y)")
    plt.title("EIA Form 930 Data for California IOUs")
    plt.legend()
    plt.grid()
    plt.gca()
    return


@app.cell
def _():
    import os
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import datetime as dt
    return dt, mo, np, pd, plt


if __name__ == "__main__":
    app.run()
