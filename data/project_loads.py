

import marimo

__generated_with = "0.13.3"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""This notebook projects load data from 2018 into 2019 through 2022 using 2018 weather sensitivity analysis and weather data for 2019 through 2022.""")
    return


@app.cell
def _(mo):
    data_ui = mo.ui.dropdown(label="Dataset:",options=["baseload","heating","cooling","total","temperature"],value="temperature")
    year_ui = mo.ui.dropdown(label="Year:",options=range(2018,2023),value=2018)
    return data_ui, year_ui


@app.cell
def _(data_ui, mo, year_ui):
    mo.hstack([data_ui,year_ui],justify='start')
    return


@app.cell
def _(data_ui, pd, year_ui):
    data = {}
    for _dataset in data_ui.options:
        data[_dataset] = {}
        for _year in [int(x) for x in year_ui.options]:
            try:
                data[_dataset][_year] = pd.read_csv(f"geodata/{_dataset}_{_year}.csv", index_col=0, date_parser=lambda x:x+"+00:00",)
            except:
                data[_dataset][_year] = pd.DataFrame([])
    return (data,)


@app.cell
def _(data, data_ui, mo, px, year_ui):
    # _fig,_ax = plt.pyplot.subplots(figsize=(15, 5))
    # _data = data[data_ui.value][year_ui.value].sum(axis=1)/1000
    # _ax.plot(_data.index,_data)
    # _ax.xaxis.set_major_locator(plt.dates.MonthLocator())
    # _ax.xaxis.set_minor_locator(plt.dates.MonthLocator(bymonthday=15))
    # _ax.xaxis.set_major_formatter(plt.ticker.NullFormatter())
    # _ax.xaxis.set_minor_formatter(plt.dates.DateFormatter('%b'))
    # _ax.tick_params(axis='x', which='minor', tick1On=False, tick2On=False)
    # for label in _ax.get_xticklabels(minor=True):
    #     label.set_horizontalalignment('center')
    # _ax.set_xlabel(str(_data.index[len(_data) // 2].year))
    # _ax.grid()
    # _ax.set_xlim([_data.index[0],_data.index[-1]])
    _ax = px.scatter(data[data_ui.value][year_ui.value].sum(axis=1)/1000)
    mo.vstack([_ax,data[data_ui.value][year_ui.value]])
    return


@app.cell
def _(data, pd, sp):
    # sensitivity analysis
    _size = min(
        [len(data[x][2018].index) for x in ["heating", "cooling", "temperature"]]
    )
    _heating = pd.DataFrame(
        data=[
            sp.linregress(
                data["temperature"][2018].iloc[:_size][x],
                data["heating"][2018].iloc[:_size][x],
                alternative="less",
            )[:2]
            for x in data["temperature"][2018].columns
        ],
        index=data["temperature"][2018].columns,
        columns=["slope","intercept"]
    ).round(3)
    _cooling = pd.DataFrame(
        data = [
            sp.linregress(
                data["temperature"][2018].iloc[:_size][x],
                data["cooling"][2018].iloc[:_size][x],
                alternative="greater",
            )[:2]
            for x in data["temperature"][2018].columns
        ],
        index=data["temperature"][2018].columns,
        columns=["slope","intercept"]
    ).round(3)
    sensitivity = _heating.join(_cooling,lsuffix="_heating",rsuffix="_cooling")
    sensitivity
    return


app._unparsable_cell(
    r"""
    # projections
    for _year in range(2019,2023):
    
    """,
    name="_"
)


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import datetime as dt
    import matplotlib as plt
    import plotly.express as px
    import scipy.stats as sp
    return mo, pd, px, sp


if __name__ == "__main__":
    app.run()
