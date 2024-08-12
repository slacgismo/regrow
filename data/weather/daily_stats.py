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
def __(mo, nodes):
    # Drop down selection for all nodes, default selection is the first node
    nodes_dropdown = mo.ui.dropdown(nodes, value=nodes[0], label='Select a node:')
    nodes_dropdown
    return nodes_dropdown,


@app.cell
def __(nodes_dropdown, pd, temperature):
    location = temperature[nodes_dropdown.value]
    location.index = location.index - pd.Timedelta(8, 'hr')
    return location,


@app.cell
def __():
    # Temperature Residual Function
    def analyze_baseline(df):
        actual = df.loc['2020-08-01':'2020-08-31'].values
        predicted = (df.loc['2018-08-01':'2018-08-31'].values 
                     + df.loc['2019-08-01':'2019-08-31'].values 
                     + df.loc['2021-08-01':'2021-08-31'].values) / 3
        return actual - predicted
    return analyze_baseline,


@app.cell
def __(mo):
    mo.md(
        r"""
        ### August 16 through 19 in 2020, excessive heat was forecasted consistently for California.
        Graphs display a slight drop in temperature followed by abnormal temperature spikes, displaying climate oscillation.
        """
    )
    return


@app.cell
def __(analyze_baseline, location, mo, plt):
    daily_residual = analyze_baseline(location.resample(rule="1D").mean())

    # August 16 through 19, excessive heat was forecasted consistently for California.
    plt.figure(figsize=(9, 5))
    plt.axvline(16, linestyle='-.',color = 'r', label = 'start of heatwave')
    plt.axvline(19, linestyle='-.',color = 'b', label = 'end of heatwave')
    # plt.axhline(0, linestyle=':',color = 'b', label = 'baseline')
    plt.plot(daily_residual)
    plt.xlabel('Days in August')
    plt.ylabel('Temperature (°C)')
    plt.title('Daily Residual Temperature')
    plt.legend()
    plt.gcf().autofmt_xdate() 
    mo.mpl.interactive(plt.gcf())
    return daily_residual,


@app.cell
def __(daily_residual, mo):
    max_daily = daily_residual.max() 
    mo.md(f"Max residual temperature: {max_daily:.2f} (C˚)")
    return max_daily,


@app.cell
def __(daily_residual, np):
    mid_point = len(daily_residual) // 2
    first_integral = np.sum(daily_residual[:mid_point])
    second_integral = np.sum(daily_residual[mid_point:])
    daily_integral = np.sum(daily_residual)
    return daily_integral, first_integral, mid_point, second_integral


@app.cell
def __(daily_integral, mo):
    mo.md(f"Overall temperature integral of August: {daily_integral:.2f} (C˚)")
    return


@app.cell
def __(first_integral, mo, second_integral):
    mo.hstack([mo.md(f"First half of August: {first_integral:.2f} (C˚),"), mo.md(f"Second half of August: {second_integral:.2f} (C˚)")], justify='start')
    return


if __name__ == "__main__":
    app.run()
