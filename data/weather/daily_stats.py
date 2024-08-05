import marimo

__generated_with = "0.7.12"
app = marimo.App(width="medium")


@app.cell
def __(mo):
    mo.md(r"""# Regrow: Weather analysis of daily temperature""")
    return


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import matplotlib.pyplot as plt
    import os, sys
    import numpy as np
    import tornado as tn
    from pathlib import Path
    import utils
    sys.path.insert(0,"..")
    return Path, mo, np, os, pd, plt, sys, tn, utils


@app.cell
def __(Path, __file__, pd):
    # Loading the Data
    _fp = Path(__file__).parent / 'temperature.csv'
    temperature = pd.read_csv(_fp, index_col=0, parse_dates=[0])
    return temperature,


@app.cell
def __(nodes, pd, utils):
    # Manipulating data from nodes to latitude/longitude
    latlong = pd.DataFrame(index=nodes, columns=['lat', 'lon'])
    for node in nodes:
        latlong.loc[node] = utils.geocode(node)
    return latlong, node


@app.cell
def __(temperature):
    nodes = temperature.columns.tolist()
    return nodes,


@app.cell
def __(mo, nodes):
    # Drop down selection for all nodes, default selection is the first node
    nodes_dropdown = mo.ui.dropdown(nodes, value=nodes[0], label='Select a node:')
    nodes_dropdown
    return nodes_dropdown,


@app.cell
def __(get_location, pd, temperature):
    location = temperature[get_location()]
    location.index = location.index - pd.Timedelta(7, 'hr')

    # Temperature Residual Function
    def analyze_baseline(df, node):
        actual = df.loc['2020-08-01':'2020-08-31'].values
        predicted = (df.loc['2018-08-01':'2018-08-31'].values 
                     + df.loc['2019-08-01':'2019-08-31'].values 
                     + df.loc['2021-08-01':'2021-08-31'].values) / 3
        return actual - predicted
    return analyze_baseline, location


if __name__ == "__main__":
    app.run()
