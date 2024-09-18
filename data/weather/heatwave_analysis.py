import marimo

__generated_with = "0.7.12"
app = marimo.App(width="medium")


@app.cell
def __(mo):
    mo.md(
        r"""
        # REGROW: Temperature Report
        Study of extreme weather and temperature rises in Western Interconnection (WECC) locations. Overlapping and comparison of temperature and solar irradance.
        """
    )
    return


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import os, sys
    import numpy as np
    from pathlib import Path
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.dates as mdates
    import tornado as tn
    sys.path.insert(0,"..")
    import utils
    return Path, mdates, mo, np, os, pd, plt, sns, sys, tn, utils


@app.cell
def __(Path, __file__, pd):
    # Loading the Data
    _tfp = Path(__file__).parent / 'temperature.csv'
    temperature = pd.read_csv(_tfp, index_col=0, parse_dates=[0])

    _sfp = Path(__file__).parent / 'solar.csv'
    solar = pd.read_csv(_tfp, index_col=0, parse_dates=[0])
    return solar, temperature


@app.cell
def __(pd, solar, temperature):
    temperature.index = temperature.index - pd.Timedelta(8, 'hr')
    solar.index = solar.index - pd.Timedelta(8, 'hr')
    return


@app.cell
def __(solar, temperature):
    nodes_temp = temperature.columns.tolist()
    nodes_solar = solar.columns.tolist()

    nodes = nodes_solar
    return nodes, nodes_solar, nodes_temp


@app.cell
def __(nodes, pd, solar, temperature):
    combined_data = pd.DataFrame(index=solar.index)

    for node in nodes:
        combined_data[f'{node}_temperature'] = temperature[node]
        combined_data[f'{node}_solar'] = solar[node]
    return combined_data, node


if __name__ == "__main__":
    app.run()
