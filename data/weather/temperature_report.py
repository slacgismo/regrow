import marimo

__generated_with = "0.7.12"
app = marimo.App(width="medium")


@app.cell
def __(mo):
    mo.md(
        r"""
        # REGROW: Temperature Report
         Study of extreme weather. Measuring the magnitude of heatwaves through temperature peaks and integrals.
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
def __(pd, temperature):
    temperature.index = temperature.index - pd.Timedelta(8, 'hr')
    return


@app.cell
def __(temperature):
    nodes = temperature.columns.tolist()
    return nodes,


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
    mo.md(r"""## Hourly Statistics""")
    return


@app.cell
def __(analyze_baseline, nodes, np, pd, temperature):
    max_residuals_hourly = []
    august_integral_hourly = []
    firsthalf_hourly = []
    secondhalf_hourly = []

    for node_hourly in nodes:
        hourly_residual = analyze_baseline(temperature[node_hourly])
        midpoint_hourly = len(hourly_residual) // 2
        max_hourly = hourly_residual.max()
        max_residuals_hourly = np.append(max_residuals_hourly, max_hourly)
        integral_one_hourly = np.sum(hourly_residual[:midpoint_hourly]) / 24
        firsthalf_hourly = np.append(firsthalf_hourly, integral_one_hourly)
        integral_two_hourly = np.sum(hourly_residual[midpoint_hourly:]) / 24
        secondhalf_hourly = np.append(secondhalf_hourly, integral_two_hourly)
        hourly_integral = np.sum(hourly_residual) / 24
        august_integral_hourly = np.append(august_integral_hourly, hourly_integral)

    hourly_report = pd.DataFrame(
        {
            "Max Hourly Residuals": max_residuals_hourly,
            "August Hourly Inegrals": august_integral_hourly,
            "First Hourly 1/2 August": firsthalf_hourly,
            "Second Hourly 1/2 August": secondhalf_hourly,
        },
        index=nodes,
    )
    return (
        august_integral_hourly,
        firsthalf_hourly,
        hourly_integral,
        hourly_report,
        hourly_residual,
        integral_one_hourly,
        integral_two_hourly,
        max_hourly,
        max_residuals_hourly,
        midpoint_hourly,
        node_hourly,
        secondhalf_hourly,
    )


@app.cell
def __(hourly_report, mo):
    mo.ui.dataframe(hourly_report)
    return


@app.cell
def __(mo):
    mo.md(r"""## Daily Statistics""")
    return


@app.cell
def __(analyze_baseline, nodes, np, pd, temperature):
    max_residuals_daily = []
    august_integral_daily = []
    firsthalf_daily = []
    secondhalf_daily = []

    for node_daily in nodes:
        daily_residual = analyze_baseline(temperature[node_daily].resample(rule="1D").mean())
        midpoint_daily = len(daily_residual) // 2
        max_daily = daily_residual.max()
        max_residuals_daily = np.append(max_residuals_daily, max_daily)
        integral_one_daily = np.sum(daily_residual[:midpoint_daily]) / 24
        firsthalf_daily = np.append(firsthalf_daily, integral_one_daily)
        integral_two_daily = np.sum(daily_residual[midpoint_daily:]) / 24
        secondhalf_daily = np.append(secondhalf_daily, integral_two_daily)
        daily_integral = np.sum(daily_residual) / 24
        august_integral_daily = np.append(august_integral_daily, daily_integral)

    daily_report = pd.DataFrame(
        {
            "Max Daily Residuals": max_residuals_daily,
            "August Daily Inegrals": august_integral_daily,
            "First Daily 1/2 August": firsthalf_daily,
            "Second Daily 1/2 August": secondhalf_daily,
        },
        index=nodes,
    )
    return (
        august_integral_daily,
        daily_integral,
        daily_report,
        daily_residual,
        firsthalf_daily,
        integral_one_daily,
        integral_two_daily,
        max_daily,
        max_residuals_daily,
        midpoint_daily,
        node_daily,
        secondhalf_daily,
    )


@app.cell
def __(daily_report, mo):
    mo.ui.dataframe(daily_report)
    return


if __name__ == "__main__":
    app.run()
