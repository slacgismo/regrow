import marimo

__generated_with = "0.11.18"
app = marimo.App(width="medium")


@app.cell
def _(aggregate_by, aggregates, fig, gendata, gentype, map_by_bus, mo):
    mo.ui.tabs({
        "Raw data" : gendata,
        "Map" : mo.vstack([mo.hstack([gentype,map_by_bus],justify='start'),fig]),
        "Aggregations" : mo.vstack([mo.hstack([gentype,aggregate_by],justify='start'),aggregates]),
    },lazy=True)
    return


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
    gendata["county"] = [utils.nearest(x, COUNTY) for x in gendata.bus]
    gendata["state"] = [COUNTY[x]["state"] for x in gendata.county]
    gendata["latitude"] = [COUNTY[x]["latitude"] for x in gendata.county]
    gendata["longitude"] = [COUNTY[x]["longitude"] for x in gendata.county]
    gendata["county"] = [COUNTY[x]["county"] for x in gendata.county]
    _bchydro_capacity = pd.read_csv(
        "../data/BCHydro/bchydro_gen.csv", index_col=["Date"]
    ).values.max().round(-2)
    _bchydro_bus = [x for x, y in COUNTY.items() if y["county"].startswith("Whatcom")][0]
    gendata = pd.concat([gendata,pd.DataFrame(
        data={
            "name": "BC_HYDRO",
            "bus": _bchydro_bus,
            "gen": "TL",
            "cap": _bchydro_capacity,
            "cf": 0.0,
            "units": 1,
            "county": "BC Hydro",
            "state": "BC",
            "latitude": 49.5,
            "longitude": -122.5,
        },index=[len(gendata)])])
    return (gendata,)


@app.cell
def _(mo):
    types = {
        "Hydroelectric": "HT",
        "Photovoltaic": "PV",
        "Steam": "ST",
        "Combustion": "CT",
        "Combined cycle": "CC",
        "Wind": "WT",
        "Internal": "IC",
        "Storage": "ES",
        "Tieline": "TL",
    }
    gentype = mo.ui.multiselect(label="Show generator types:",options=types,value=list(types))
    return gentype, types


@app.cell
def _(gendata, gentype, map_by_bus, px, utils):
    if map_by_bus.value:
        _data = (
            gendata[gendata.gen.isin(gentype.value)]
            .groupby(["latitude", "longitude", "bus"])
            .sum()[["cap", "cf", "units"]]
            .round(1)
            .reset_index()
        )
        _data[["latitude","longitude"]] = [[lat,lon] for lat,lon in [utils.geocode(x) for x in _data.bus]]
        fig = px.scatter_map(
            _data,
            lat="latitude",
            lon="longitude",
            size="cap",
            color="units",
            zoom=4.2,
            hover_name="bus",
            hover_data={"latitude":False,"longitude":False,"cap":True,"units":True},
            width = 800,height = 800,
            center = {"lat":41,"lon":-114},
        )
    else:
        _data = (
            gendata[gendata.gen.isin(gentype.value)]
            .groupby(["latitude", "longitude", "county","state"])
            .sum()[["cap", "cf", "units"]]
            .round(1)
            .reset_index()
        )
        fig = px.scatter_map(
            _data,
            lat="latitude",
            lon="longitude",
            size="cap",
            color="units",
            zoom=4.2,
            hover_name=[f"{x} {y}" for x,y in _data[["county","state"]].values],
            hover_data={"latitude":False,"longitude":False,"cap":True,"units":True},
            width = 800,height = 800,
            center = {"lat":41,"lon":-114},
        )

    return (fig,)


@app.cell
def _(mo):
    map_by_bus = mo.ui.checkbox(label="Show busses")
    return (map_by_bus,)


@app.cell
def _(mo):
    aggregate_by = mo.ui.dropdown(label="Aggregate by",options={"bus":"bus","generator type":"gen","county":["county","state"]},value="bus")
    return (aggregate_by,)


@app.cell
def _(aggregate_by, gendata, gentype):
    aggregates = gendata[gendata.gen.isin(gentype.value)].groupby(aggregate_by.value).sum().round(1)[["cap","cf","units"]]
    return (aggregates,)


@app.cell
def _():
    import os
    import sys
    import marimo as mo
    import pandas as pd
    import plotly.express as px
    import matplotlib.pyplot as plt
    mo.stop(not "GLD_ETC" in os.environ, mo.md("ERROR [check_powerplants.py]: not running in a gridlabd container/environment"))
    import gridlabd.census as census
    sys.path.append("../data")
    import utils
    return census, mo, os, pd, plt, px, sys, utils


if __name__ == "__main__":
    app.run()
