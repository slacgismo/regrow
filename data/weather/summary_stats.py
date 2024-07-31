import marimo

__generated_with = "0.7.12"
app = marimo.App(width="medium")


@app.cell
def __(mo):
    mo.md("""# REGROW: Magnitude of Heat Waves""")
    return


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import os, sys
    import numpy as np
    import tornado as tn
    from pathlib import Path
    sys.path.insert(0,"..")
    import utils
    return Path, mdates, mo, np, os, pd, plt, sys, tn, utils


@app.cell
def __(Path, __file__, pd):
    # Loading the Data
    _fp = Path(__file__).parent / 'temperature.csv'
    temperature = pd.read_csv(_fp, index_col=0, parse_dates=[0])
    return temperature,


@app.cell
def __():
    # Geographic location encoding/decoding
    # _utils = Path(__file__).parent / 'utils.py'
    return


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
def __(mo):
    mo.md("""## Viewing nodes on Google Earth:""")
    return


@app.cell
def __(Path, __file__, mo):
    # Google Earth Snapshot
    _img = (Path(__file__).parent / 'wecc_google_earth.png')
    mo.image(src=f"{_img}")
    return


@app.cell
def __(mo, nodes, os, pd, utils):
    # Converting geohash list into a dropdown that includes county names
    get_location, set_location = mo.state(nodes[0])
    _counties = pd.read_csv(os.path.join("..","counties.csv"),index_col="geocode")
    _options = dict([(f"{x} ({_counties.loc[utils.nearest(x,_counties.index)].county})",x) for x in nodes])
    _index = dict([(y,x) for x,y in _options.items()])

    # Drop down selection for all nodes, default selection is the first node
    location_ui = mo.ui.dropdown(
        label="Location:",
        on_change=set_location,
        options=_options, # locations,
        value=_index[get_location()],
        allow_select_none=False,
    )

    # Toggle between daily to hourly average temps
    get_daily_switch,set_daily_switch = mo.state(False)
    grouping_switch = mo.ui.switch(label="Hourly | Daily Average",value=get_daily_switch(),on_change=set_daily_switch)
    mo.hstack([location_ui,grouping_switch],justify='start')
    return (
        get_daily_switch,
        get_location,
        grouping_switch,
        location_ui,
        set_daily_switch,
        set_location,
    )


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


@app.cell
def __(mo):
    mo.md(
        """
        ### August 16 through 19 in 2020, excessive heat was forecasted consistently for California.
        Graphs display a slight drop in temperature followed by abnormal temperature spikes, displaying climate oscillation.
        """
    )
    return


@app.cell
def __(
    analyze_baseline,
    get_daily_switch,
    get_location,
    location,
    mo,
    plt,
):
    def _is_daily(x,y):
        return x if get_daily_switch() else y

    daily_residual = analyze_baseline(location.resample(rule=_is_daily("1D","1h")).mean(), get_location())
    hourly_residual = analyze_baseline(location, get_location())

    plt.figure(figsize=(9, 5))

    # August 16 through 19, excessive heat was forecasted consistently for California.
    plt.axvline(_is_daily(16, 16*24), linestyle='--',color = 'r', label = 'start of heatwave')
    plt.axvline(_is_daily(19, 19*24), linestyle='--',color = 'b', label = 'end of heatwave')
    plt.axhline(0,0,daily_residual.shape[0],linestyle=':',label='Baseline')
    plt.plot(daily_residual)
    plt.xlabel(_is_daily('Days in August','Hours in August'))
    plt.ylabel(_is_daily('Average Temperature (°C)', 'Temperature (°C)'))
    plt.title(_is_daily('Daily Residual Temperature', 'Hourly Residual Temperature'))
    plt.legend()
    plt.gcf().autofmt_xdate() 
    mo.mpl.interactive(plt.gcf())
    return daily_residual, hourly_residual


@app.cell
def __():
    # # Toggle between daily to hourly max deviations
    # get_max_deviation,deviation_switch = mo.state(False)
    # grouping_deviation_switch = mo.ui.switch(label="Hourly / Daily deviations",value=max_daily_deviation(),on_change=max_hourly_deviation)
    return


@app.cell
def __():
    # def _max_deviation(x,y):
    #     return x if deviation_switch() else y

    # daily_std = daily_residual.std()
    # max_daily_deviation = daily_std.max().round(3)

    # hourly_std = hourly_residual.std()
    # max_hourly_deviation = hourly_residual.max().round(3)

    # mo.md(_max_deviation(f"Max Daily Deviation: {max_daily_deviation}", f"Max Hourly Deviation: {max_hourly_deviation}"))
    return


@app.cell
def __(daily_residual, mo):
    daily_std = daily_residual.std()
    max_daily_deviation = daily_std.max().round(3)

    mo.md(f"Max Daily Deviation: {max_daily_deviation}")
    return daily_std, max_daily_deviation


@app.cell
def __(hourly_residual, mo):
    hourly_std = hourly_residual.std()
    max_hourly_deviation = hourly_residual.max().round(3)

    mo.md(f"Max Hourly Deviation: {max_hourly_deviation}")
    return hourly_std, max_hourly_deviation


@app.cell
def __():
    # plt.hist(hourly_residual)
    return


if __name__ == "__main__":
    app.run()
