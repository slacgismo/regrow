import marimo

__generated_with = "0.7.12"
app = marimo.App(width="medium")


@app.cell
def __(mo):
    mo.md(
        r"""
        # REGROW: Temperature Report
        Study of extreme weather and temperature rises in Western Interconnection (WECC) locations. Report measures the magnitude of the 2020 heatwave through max residual temperature and integrals.
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
def __(Path, __file__, mo):
    # Google Earth Snapshot
    _img = (Path(__file__).parent / 'wecc_google_earth.png')
    mo.image(src=f"{_img}")
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Plotting the Data
        Full time series plot from years 2018-2022.
        """
    )
    return


@app.cell
def __(mo, nodes, os, pd, utils):
    # Converting geohash list into a dropdown that includes county names
    get_location, set_location = mo.state(nodes[0])
    _counties = pd.read_csv(os.path.join("..","counties.csv"),index_col="geocode")
    _options = dict([(f"{x} ({_counties.loc[utils.nearest(x,_counties.index)].county})",x) for x in nodes])
    _index = dict([(y,x) for x,y in _options.items()])
    location_ui = mo.ui.dropdown(
        label="Location:",
        on_change=set_location,
        options=_options, # locations,
        value=_index[get_location()],
        allow_select_none=False,
    )
    location_ui
    return get_location, location_ui, set_location


@app.cell
def __(location_ui, temperature):
    location = temperature[location_ui.value]
    return location,


@app.cell
def __(heat_map, mo, time_series):
    mo.hstack([time_series, heat_map])
    return


@app.cell
def __(get_location, plt, temperature):
    # Time Series of Temperatures (2018-2022)
    data_view = temperature[get_location()]
    data_view.plot()
    plt.xlabel('Year')
    plt.ylabel('Average Temperature (C˚)')
    plt.title('Temperature (2018-2022)')
    time_series = plt.gcf()
    return data_view, time_series


@app.cell
def __(data_view, plt, sns):
    # Heat Map
    my_data_array = data_view.loc['2018-01-01':'2022-12-30'].values.reshape((24, -1), order='F')
    sns.heatmap(my_data_array, cmap="plasma")
    plt.xlabel('Days')
    plt.ylabel('Hours')
    plt.title('Heat map of temperatures (2018-2022)')
    heat_map = plt.gcf()
    return heat_map, my_data_array


@app.cell
def __(location, pd):
    # Time slicing for August
    august1 = location.loc['2018-08-01':'2018-08-31']
    august2 = location.loc['2019-08-01':'2019-08-31']
    august3 = location.loc['2020-08-01':'2020-08-31']
    august4 = location.loc['2021-08-01':'2021-08-31']
    august5 = location.loc['2022-08-01':'2022-08-31']

    august1 = pd.DataFrame(august1)
    august2 = pd.DataFrame(august2)
    august3 = pd.DataFrame(august3)
    august4 = pd.DataFrame(august4)
    august5 = pd.DataFrame(august5)
    return august1, august2, august3, august4, august5


@app.cell
def __(august1, august2, august3, august4, august5, mo, plt):
    # Calculated average daily temperatures
    avg_daily_2018 = august1.resample(rule="1D").mean()
    avg_daily_2019 = august2.resample(rule="1D").mean()
    avg_daily_2020 = august3.resample(rule="1D").mean()
    avg_daily_2021 = august4.resample(rule="1D").mean()
    avg_daily_2022 = august5.resample(rule="1D").mean()

    # Plotting the data
    plt.figure(figsize=(9, 5))
    plt.plot(avg_daily_2018.values, label='2018')
    plt.plot(avg_daily_2019.values, label='2019')
    plt.plot(avg_daily_2020.values, label='2020', ls=":")
    plt.plot(avg_daily_2021.values, label='2021')
    plt.plot(avg_daily_2022.values, label='2022')

    plt.xlabel('Date')
    plt.ylabel('Average Temperature (C˚)')
    plt.title('Daily Average Temperature (August 2018-2022)')
    plt.legend()
    plt.gcf().autofmt_xdate() 
    mo.mpl.interactive(plt.gcf())
    return (
        avg_daily_2018,
        avg_daily_2019,
        avg_daily_2020,
        avg_daily_2021,
        avg_daily_2022,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Hourly Statistics
        August 16 through 19 in 2020, excessive heat was forecasted consistently for California.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(r"""To calculate the residual temperature of 2020, the median of the surroundnig years was taken to create a baseline for comparison.""")
    return


@app.cell
def __(np):
    # Temperature Residual Function
    def analyze_baseline(df):
        actual = df.loc['2020-08-01':'2020-08-31'].values
        predicted = np.c_[
            df.loc['2018-08-01':'2018-08-31'].values, 
            df.loc['2019-08-01':'2019-08-31'].values,
            df.loc['2021-08-01':'2021-08-31'].values,
            df.loc['2022-08-01':'2022-08-31'].values
        ]
        predicted = np.median(predicted, axis=1)
        return actual - predicted
    return analyze_baseline,


@app.cell
def __(analyze_baseline, location, mo, plt):
    hourly_residual = analyze_baseline(location)

    # August 16 through 19, excessive heat was forecasted consistently for California.
    plt.figure(figsize=(9, 5))
    plt.axvline(16 * 24, linestyle='-.',color = 'r', label = 'start of heatwave')
    plt.axvline(19 * 24, linestyle='-.',color = 'b', label = 'end of heatwave')
    plt.axhline(0, linestyle=':',color = 'b', label = 'baseline')
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


@app.cell
def __(mo):
    mo.md(r"""Graphs display a slight drop in temperature followed by abnormal temperature spikes, displaying climate oscillation.""")
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Full Report
        (1) Max residual temperature. (2) August integrals 1st half, 2nd half and overall.
        """
    )
    return


@app.cell
def __(analyze_baseline, nodes, np, pd, temperature):
    # Initial lists for results
    max_residuals = []
    august_integral = []
    firsthalf = []
    secondhalf = []

    # Calculates residuals and integrals for each node
    for node in nodes:
        residual = analyze_baseline(temperature[node])
        midpoint = len(residual) // 2
        max = residual.max()
        max_residuals = np.append(max_residuals, max)

        integral_one = np.sum(residual[:midpoint]) / 24
        firsthalf = np.append(firsthalf, integral_one)

        integral_two = np.sum(residual[midpoint:]) / 24
        secondhalf = np.append(secondhalf, integral_two)

        integral = np.sum(residual) / 24
        august_integral = np.append(august_integral, integral)

    # DataFrame with results
    report = pd.DataFrame(
        {
            "Max Hourly Residuals": max_residuals,
            "August Hourly Inegrals": august_integral,
            "First Hourly 1/2 August": firsthalf,
            "Second Hourly 1/2 August": secondhalf,
        },
        index=nodes,
    )
    return (
        august_integral,
        firsthalf,
        integral,
        integral_one,
        integral_two,
        max,
        max_residuals,
        midpoint,
        node,
        report,
        residual,
        secondhalf,
    )


@app.cell
def __(mo, report):
    mo.ui.dataframe(report.round(2))
    return


if __name__ == "__main__":
    app.run()
