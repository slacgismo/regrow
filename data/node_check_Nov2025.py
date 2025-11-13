import marimo

__generated_with = "0.17.7"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import utils
    return mo, np, pd, utils


@app.cell
def _(pd, utils):
    network = pd.read_csv("wecc240_gis.csv", 
                          usecols=["Bus  Number","Bus  Name","Lat","Long"])
    network['geohash'] = network.apply(lambda row: utils.geohash(row['Lat'], row['Long']), axis=1)
    return (network,)


@app.cell
def _(grouped):
    grouped['Bus  Number'].apply(list)
    return


@app.cell
def _(grouped):
    grouped['Bus  Name'].apply(list)
    return


@app.cell
def _(network):
    grouped = network.groupby('geohash')
    reduced_network = grouped.first()
    reduced_network['node count'] = grouped.count()['Bus  Number'].values
    return grouped, reduced_network


@app.cell
def _(reduced_network):
    reduced_network
    return


@app.cell
def _(utils):
    utils.distance('9mtzm4', '9muccv')/1e3
    return


@app.cell
def _(utils):
    utils.geocode('9mtzm4')
    return


@app.cell
def _(utils):
    utils._decode('9mtzm4')
    return


@app.cell
def _(utils):
    utils.distance('9mtzm4', '9mudw2')
    return


@app.cell
def _():
    test_lat = 38.357664
    test_lon = -121.111278
    return test_lat, test_lon


@app.cell
def _(reduced_network):
    latlon_list = [
        (reduced_network.iloc[_ix]['Lat'], reduced_network.iloc[_ix]['Long']) 
        for _ix in range(len(reduced_network))
    ]
    return (latlon_list,)


@app.cell
def _(latlon_list, test_lat, test_lon, utils):
    utils.nearest2((test_lat, test_lon), latlon_list)
    return


@app.cell
def _(reduced_network):
    reduced_network.iloc[50]
    return


@app.cell
def _(mo, np, reduced_network, test_lat, test_lon, utils):
    best_ix = 0
    best_dist = np.inf
    for _ix in mo.status.progress_bar(range(len(reduced_network))):
        _lat, _lon = reduced_network.iloc[_ix][['Lat', 'Long']].values
        _new_dist = utils.haversine_distance(_lat, _lon, test_lat, test_lon)
        if _new_dist < best_dist:
            best_dist = _new_dist
            best_ix = _ix
            print(f"new closest found!: {reduced_network.iloc[_ix].name}, {reduced_network.iloc[_ix]['Bus  Name']}")
    return


if __name__ == "__main__":
    app.run()
