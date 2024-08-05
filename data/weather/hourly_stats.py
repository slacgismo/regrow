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
def __(nodes, pd, utils):
    # Manipulating data from nodes to latitude/longitude
    latlong = pd.DataFrame(index=nodes, columns=['lat', 'lon'])
    for node in nodes:
        latlong.loc[node] = utils.geocode(node)
    return latlong, node


@app.cell
def __(mo, nodes):
    # Drop down selection for all nodes, default selection is the first node
    nodes_dropdown = mo.ui.dropdown(nodes, value=nodes[0], label='Select a node:')
    nodes_dropdown
    return nodes_dropdown,


@app.cell
def __(nodes_dropdown, pd, temperature):
    location = temperature[nodes_dropdown.value]
    location.index = location.index - pd.Timedelta(7, 'hr')

    # Temperature Residual Function
    def analyze_baseline(df, node):
        actual = df.loc['2020-08-01':'2020-08-31'].values
        predicted = (df.loc['2018-08-01':'2018-08-31'].values 
                     + df.loc['2019-08-01':'2019-08-31'].values 
                     + df.loc['2021-08-01':'2021-08-31'].values) / 3
        return actual - predicted
    return analyze_baseline, location


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
def __(analyze_baseline, location, mo, nodes_dropdown, plt):
    hourly_residual = analyze_baseline(location, nodes_dropdown.value)

    # August 16 through 19, excessive heat was forecasted consistently for California.
    plt.figure(figsize=(9, 5))
    plt.axvline(16 * 24, linestyle='-.',color = 'r', label = 'start of heatwave')
    plt.axvline(19 * 24, linestyle='-.',color = 'b', label = 'end of heatwave')
    plt.axhline(0,0,hourly_residual.shape[0],linestyle=':',label='Baseline')
    plt.plot(hourly_residual)
    plt.xlabel('Hours in August')
    plt.ylabel('Temperature (°C)')
    plt.title('Hourly Residual Temperature')
    plt.legend()
    plt.gcf().autofmt_xdate() 
    mo.mpl.interactive(plt.gcf())
    return hourly_residual,


@app.cell
def __(hourly_residual, mo):
    max_hourly = hourly_residual.max() 
    mo.md(f"Max residual temperature: {max_hourly:.2f} (C˚)")
    return max_hourly,


@app.cell
def __(hourly_residual, np):
    mid_point = len(hourly_residual) // 2
    first_integral = np.sum(hourly_residual[:mid_point]) / 24 
    second_integral = np.sum(hourly_residual[mid_point:]) / 24
    hourly_integral = np.sum(hourly_residual) / 24
    return first_integral, hourly_integral, mid_point, second_integral


@app.cell
def __(hourly_integral, mo):
    mo.md(f"Overall temperature integral of August: {hourly_integral:.2f} (C˚)")
    return


@app.cell
def __(first_integral, mo, second_integral):
    mo.hstack([mo.md(f"First half of August: {first_integral:.2f} (C˚),"), mo.md(f"Second half of August: {second_integral:.2f} (C˚)")], justify='start')
    return


if __name__ == "__main__":
    app.run()
