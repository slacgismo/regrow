import marimo

__generated_with = "0.11.18"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""## WECC 240 Bus Load vs. Generation""")
    return


@app.cell
def _(pd):
    load = pd.read_csv("loads.csv",index_col=["geocode"],usecols=["geocode","power[MVA]"]).groupby(level=0).max()
    gens = pd.read_csv("powerplants_aggregated.csv",index_col=["bus"],usecols=["bus","cap","cf"]).groupby(level=0)[["cap","cf"]].sum().round(3)
    counties = pd.read_csv("../data/counties.csv",index_col=["geocode"],usecols=["geocode","usps","county"])
    counties["county"] = [f"{x.replace(' County','').strip()} {y}" for x,y in counties[["county","usps"]].values]
    counties.drop("usps",axis=1,inplace=True)
    return counties, gens, load


@app.cell
def _(counties, gens, load, utils):
    data = load.join(gens)
    data.rename({"power[MVA]":"demand[MW]"},axis=1,inplace=True)
    data["generation[MW]"] = data["cap"].round(1)
    data["dispatchable[MW]"] = (data["cap"] * data["cf"]).round(1)
    data["renewable[MW]"] = (data["generation[MW]"] - data["dispatchable[MW]"]).round(1)
    data["imports[MW]"] = (data["demand[MW]"] - data["dispatchable[MW]"]).round(1)
    data.drop(["cap","cf"],inplace=True,axis=1)
    data["county"] = [counties.loc[utils.nearest(x,counties.index)].county for x in data.index]
    data.dropna(inplace=True)
    data["state"] = [x[-2:] for x in data["county"]]
    return (data,)


@app.cell
def _(data, px):
    _data = data.loc[data["state"]=="CA"]
    px.scatter(_data,x="renewable[MW]",y="demand[MW]",hover_name=_data.index,hover_data=["county","demand[MW]","dispatchable[MW]","generation[MW]","imports[MW]"])
    return


@app.cell
def _(data):
    data
    return


@app.cell
def _():
    import sys
    sys.path.append("../data")
    import utils
    import marimo as mo
    import pandas as pd
    import plotly.express as px
    return mo, pd, px, sys, utils


if __name__ == "__main__":
    app.run()
