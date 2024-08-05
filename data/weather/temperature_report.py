import marimo

__generated_with = "0.7.12"
app = marimo.App(width="medium")


@app.cell
def __(mo):
    mo.md(
        r"""
        # REGROW: Temperature Report
         Study of extreme weather by measuring magnitude of heatwaves through temperature peaks (max residual) and integrals.
        """
    )
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
    return Path, mo, np, os, pd, plt, sys, tn, utils


@app.cell
def __(Path, __file__, pd):
    # Loading the Data
    _fp = Path(__file__).parent / 'temperature.csv'
    temperature = pd.read_csv(_fp, index_col=0, parse_dates=[0])
    return temperature,


@app.cell
def __(temperature):
    nodes = temperature.columns.tolist()
    return nodes,


@app.cell
def __():
    # Temperature Residual Function
    def analyze_baseline(df, node): # remove second arguement
        actual = df.loc['2020-08-01':'2020-08-31'].values
        predicted = (df.loc['2018-08-01':'2018-08-31'].values 
                     + df.loc['2019-08-01':'2019-08-31'].values 
                     + df.loc['2021-08-01':'2021-08-31'].values) / 3
        return actual - predicted
    return analyze_baseline,


if __name__ == "__main__":
    app.run()
