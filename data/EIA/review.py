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
def _(pd):
    data = (pd.read_csv("eia_f861.csv", index_col=[1, 0]) / 1000000).round(4).sort_index()
    data.columns = [x[0:3].upper() for x in data.columns]
    # mo.ui.table(data,page_size=len(data.index.get_level_values(1).unique()),selection=None)
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
    growth = (
        data.loc[state_ui.value,:]
        / data.loc[state_ui.value,2018]
    )
    # print(growth)
    growth.plot(
        title=f"EIA Form 861 Data for {state_ui.value}",
        grid=True,
        xlabel="Year",
        ylabel="Load growth (%/y)",
        xticks=data.index.get_level_values(1).unique().to_list(),
        legend="outside",
    )
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
    _data = demand.loc[region_ui.value,utility_ui.value]/1e3
    _data.plot(figsize=(10,5),grid=True,xlabel="Date",ylabel="GW",legend=False)
    return


@app.cell
def _():
    import os
    import marimo as mo
    import pandas as pd
    return mo, pd


if __name__ == "__main__":
    app.run()
