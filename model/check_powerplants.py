import marimo

__generated_with = "0.10.17"
app = marimo.App(width="medium")


@app.cell
def _(census, pd, utils):
    COUNTY = {}
    for _state in ["CA", "WA", "OR", "ID", "MT", "WY", "NV", "UT", "AZ", "NM", "CO","TX","SD"]:
        _fips = f"{census.FIPS_STATES[_state]['fips']:02.0f}"

        # County population centroid data
        _url = f"https://www2.census.gov/geo/docs/reference/cenpop2020/county/CenPop2020_Mean_CO{_fips}.txt"
        _df = pd.read_csv(
            _url,
            converters={
                "COUNAME": census.strict_ascii,
                "STATEFP": lambda x: f"{float(x):02.0f}",
                "COUNTYFP": lambda x: f"{float(x):03.0f}",
            },
            usecols=[
                "STATEFP",
                "COUNTYFP",
                "COUNAME",
                "LATITUDE",
                "LONGITUDE",
            ],
        )
        _df["STATE"] = _state
        _df["GEOCODE"] = [utils.geohash(x,y) for x,y in zip(_df.LATITUDE,_df.LONGITUDE)]
        _df.drop(["STATEFP","COUNTYFP"],inplace=True,axis=1)
        _df.columns = ["county","latitude","longitude","state","geocode"]
        COUNTY.update(_df.set_index("geocode").to_dict('index'))
    return (COUNTY,)


@app.cell
def _(COUNTY, pd, utils):
    gendata = pd.read_csv("powerplants_data.csv")
    gendata["county"] = [utils.nearest(x,COUNTY) for x in gendata.bus]
    gendata["state"] = [COUNTY[x]["state"] for x in gendata.county]
    gendata["latitude"] = [COUNTY[x]["latitude"] for x in gendata.county]
    gendata["longitude"] = [COUNTY[x]["longitude"] for x in gendata.county]
    gendata["county"] = [COUNTY[x]["county"] for x in gendata.county]
    return (gendata,)


@app.cell
def _(mo):
    mo.md(r"""## Powerplant Data""")
    return


@app.cell
def _(gendata):
    gendata
    return


@app.cell
def _(gendata, px):
    _data = gendata.groupby(["bus","latitude","longitude","county"]).sum()[["cap","cf","units"]].reset_index()
    fig = px.scatter_map(_data,lat="latitude",lon="longitude",size="cap",color="units",zoom=3,hover_name="county")
    fig

    return (fig,)


@app.cell
def _(mo):
    by = mo.ui.dropdown(options={"bus":"bus","generator type":"gen","county":["county","state"]},value="bus")
    return (by,)


@app.cell
def _(by, mo):
    mo.md(f"""Powerplants by {by}""")
    return


@app.cell
def _(by, gendata):
    gendata.groupby(by.value).sum().round(1)[["cap","cf","units"]]
    return


@app.cell
def _():
    import os
    import sys
    import marimo as mo
    import pandas as pd
    import plotly.express as px
    import gridlabd.census as census
    sys.path.append("../data")
    import utils
    return census, mo, os, pd, px, sys, utils


if __name__ == "__main__":
    app.run()
