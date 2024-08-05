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
    import sympy as sy
    return Path, mdates, mo, np, os, pd, plt, sy, sys, tn, utils


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
    grouping_switch = mo.ui.switch(label="Hourly | Daily Average")
    mo.hstack([location_ui,grouping_switch],justify='start')
    return get_location, grouping_switch, location_ui, set_location


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
    plt.axvline(_is_daily(16, 16*24), linestyle='-.',color = 'r', label = 'start of heatwave')
    plt.axvline(_is_daily(19, 19*24), linestyle='-.',color = 'b', label = 'end of heatwave')
    plt.axhline(0,0,daily_residual.shape[0],linestyle=':',label='Baseline')
    plt.plot(daily_residual)
    plt.xlabel(_is_daily('Days in August','Hours in August'))
    plt.ylabel(_is_daily('Temperature (°C)', 'Temperature (°C)'))
    plt.title(_is_daily('Daily Residual Temperature', 'Hourly Residual Temperature'))
    plt.legend()
    plt.gcf().autofmt_xdate() 
    mo.mpl.interactive(plt.gcf())
    return daily_residual, hourly_residual


@app.cell
def __(daily_residual, hourly_residual, mo):
    max_daily = daily_residual.max() 
    max_hourly = hourly_residual.max() 
    mo.md(f"Max residual temperature: {max_daily:.2f} (C˚)")
    return max_daily, max_hourly


@app.cell
def __(mo):
    mo.md(r"""### Temperature Integrals""")
    return


@app.cell
def __(daily_residual, hourly_residual, np):
    # First half of August
    first_daily_integral = np.sum(daily_residual[:15])
    first_hourly_integral = np.sum(hourly_residual[:360]) / 24

    # Second half of August
    second_daily_integral = np.sum(daily_residual[15:])
    second_hourly_integral = np.sum(hourly_residual[360:]) / 24
    return (
        first_daily_integral,
        first_hourly_integral,
        second_daily_integral,
        second_hourly_integral,
    )


@app.cell
def __(first_daily_integral, mo, second_daily_integral):
    mo.hstack([(mo.md(f"Daily first half of August: {first_daily_integral:.2f} (C˚),")),mo.md(f"Daily second half of August: {second_daily_integral:.2f} (C˚)")],justify='start')
    return


@app.cell
def __(first_hourly_integral, mo, second_hourly_integral):
    mo.hstack([(mo.md(f"Hourly first half of August: {first_hourly_integral:.2f} (C˚),")),mo.md(f"Hourly second half of August: {second_hourly_integral:.2f} (C˚)")],justify='start')
    return


@app.cell
def __(daily_residual, hourly_residual, mo, np):
    # Final integral temperatures of August
    daily_integral = np.sum(daily_residual)
    hourly_integral = np.sum(hourly_residual) /24

    mo.hstack([(mo.md(f"Final daily temperature integral: {daily_integral:.2f} (C˚),")),mo.md(f"Final hourly temperature integral: {hourly_integral:.2f} (C˚)")],justify='start')
    return daily_integral, hourly_integral


@app.cell
def __():
    # daily_std = daily_residual.std()
    # max_daily_deviation = daily_std.max().round(3)

    # mo.md(f"Max Daily Deviation: {max_daily_deviation}")
    return


@app.cell
def __():
    # hourly_std = hourly_residual.std()
    # max_hourly_deviation = hourly_residual.max().round(3)

    # mo.md(f"Max Hourly Deviation: {max_hourly_deviation}")
    return


@app.cell
def __():
    # Calculating the Intergral
    return


@app.cell
def __():
    # plt.hist(hourly_residual)
    return


if __name__ == "__main__":
    app.run()
