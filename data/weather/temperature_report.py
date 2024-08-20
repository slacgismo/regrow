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
    import geopandas
    import geodatasets
    from geodatasets import get_path
    from matplotlib import colormaps
    return (
        Path,
        colormaps,
        geodatasets,
        geopandas,
        get_path,
        mdates,
        mo,
        np,
        os,
        pd,
        plt,
        sns,
        sys,
        tn,
        utils,
    )


@app.cell
def __(Path, __file__, pd):
    # Loading the Data
    _fp = Path(__file__).parent / 'temperature.csv'
    temperature = pd.read_csv(_fp, index_col=0, parse_dates=[0])
    return temperature,


@app.cell
def __(pd, temperature):
    # Adjusting for time zone differences (shifting by 8 hours)
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
def __(analyze_baseline, nodes, np, os, pd, temperature, utils):
    # Initial lists for results
    max_residuals = []
    august_integral = []
    firsthalf = []
    secondhalf = []

    # County data connected to correlating node
    _counties = pd.read_csv(os.path.join("..","counties.csv"),index_col="geocode")
    node_county_map = {node: _counties.loc[utils.nearest(node, _counties.index)].county for node in nodes}
    report_index = [f"{node} ({node_county_map[node]})" for node in nodes]

    # Calculates residuals and integrals for each node
    for node in nodes:
        residual = analyze_baseline(temperature[node])
        midpoint = len(residual) // 2
        _max = residual.max()
        max_residuals = np.append(max_residuals, _max)

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
        index=report_index,
    )
    return (
        august_integral,
        firsthalf,
        integral,
        integral_one,
        integral_two,
        max_residuals,
        midpoint,
        node,
        node_county_map,
        report,
        report_index,
        residual,
        secondhalf,
    )


@app.cell
def __(mo, report):
    mo.ui.dataframe(report.round(2))
    return


@app.cell
def __(mo):
    mo.md(r"""## Locational Plotting""")
    return


@app.cell
def __(geocode, nodes, pd):
    # Manipulating data from nodes to latitude/longitude
    latlong = pd.DataFrame(index=nodes, columns=['lat', 'lon'])
    for geo_node in nodes:
        latlong.loc[geo_node] = geocode(geo_node)
    return geo_node, latlong


@app.cell
def __(geopandas, latlong):
    # Geopandas library for plotting lat long to geometry shap
    gdf = geopandas.GeoDataFrame(
        latlong, geometry=geopandas.points_from_xy(latlong.lon, latlong.lat), crs="EPSG: 4326"
    )
    print(gdf)
    return gdf,


@app.cell
def __(geopandas):
    # File containing counties and state boundaries, update to only state, maybe check geopandas
    map = geopandas.read_file("c_05mr24.shp")
    print(map)
    return map,


@app.cell
def __(map):
    map.plot()
    return


@app.cell
def __(np):
    def removeRepeats(arr):
        newArr = []
        addVal = True
        for value in arr:
            for n in newArr:
                if value == n:
                    addVal = False
            if addVal:
                newArr = np.append(newArr,value)
            addVal = True
        return(newArr)
    return removeRepeats,


@app.cell
def __(map, removeRepeats):
    states = removeRepeats(map.STATE)
    return states,


@app.cell
def __(map):
    # Dataframe excluding states to narrow geographical display
    nonwestcoast = ['ME', 'GA', 'AS', 'PR', 'VI', 'CT', 'MA', 'MD', 'VT', 'NH', 'NY', 'PA', 'RI', 'VA','WV', 'NJ', 'KY', 'MI', 'MS', 'IA', 'IL', 'SD', 'KS', 'IN', 'NC', 'OH',
     'SC', 'TN', 'AL', 'NE', 'ND', 'LA', 'AR', 'MN', 'MO', 'WI', 'FL', 'TX',
     'OK','DC', 'DE', 'HI', 'PW', 'MH', 'MP', 'GU', 'FM', 'AK']
    us49 = map
    for i in nonwestcoast:
        us49 = us49[us49.STATE != i]
    return i, nonwestcoast, us49


@app.cell
def __():
    # world = geopandas.read_file(get_path("naturalearth.land"))

    # # Creating frame for West Coast plots, outline of US, location of nodes
    # ax = world.clip([-130, 30, -102, 51]).plot(color="white", edgecolor="black")
    # gdf.plot(ax=ax)
    return


@app.cell
def __():
    # # County lines
    # us49.boundary.plot()
    return


@app.cell
def __(gdf, geopandas, get_path, plt, us49):
    # Combining both
    world = geopandas.read_file(get_path("naturalearth.land"))

    # Creating frame for West Coast plots
    # ax = world.clip([-130, 30, -102, 51]).plot(color="white", edgecolor="black")
    f, ax = plt.subplots()

    us49.boundary.plot(ax=ax,color="grey")
    gdf.plot(ax=ax,color='blue')
    plt.show()
    return ax, f, world


@app.cell
def __(E_INVAL, dt, error, json, math, os, pd, pvlib_psm3, warning):
    #
    # Geographic location encoding/decoding
    #
    _cache = {}

    def _decode(geohash):
        """
        Decode the geohash to its exact values, including the error
        margins of the result.  Returns four float values: latitude,
        longitude, the plus/minus error for latitude (as a positive
        number) and the plus/minus error for longitude (as a positive
        number).
        """
        __base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
        __decodemap = { }
        for i in range(len(__base32)):
            __decodemap[__base32[i]] = i
        del i
        lat_interval, lon_interval = (-90.0, 90.0), (-180.0, 180.0)
        lat_err, lon_err = 90.0, 180.0
        is_even = True
        for c in geohash:
            cd = __decodemap[c]
            for mask in [16, 8, 4, 2, 1]:
                if is_even: # adds longitude info
                    lon_err /= 2
                    if cd & mask:
                        lon_interval = ((lon_interval[0]+lon_interval[1])/2, lon_interval[1])
                    else:
                        lon_interval = (lon_interval[0], (lon_interval[0]+lon_interval[1])/2)
                else:      # adds latitude info
                    lat_err /= 2
                    if cd & mask:
                        lat_interval = ((lat_interval[0]+lat_interval[1])/2, lat_interval[1])
                    else:
                        lat_interval = (lat_interval[0], (lat_interval[0]+lat_interval[1])/2)
                is_even = not is_even
        lat = (lat_interval[0] + lat_interval[1]) / 2
        lon = (lon_interval[0] + lon_interval[1]) / 2
        return lat, lon, lat_err, lon_err

    def geocode(geohash):
        """
        Decode geohash, returning two strings with latitude and longitude
        containing only relevant digits and with trailing zeroes removed.
        """
        if geohash in _cache:
            return _cache[geohash][0],_cache[geohash][1]
        lat, lon, lat_err, lon_err = _decode(geohash)
        from math import log10
        # Format to the number of decimals that are known
        lats = "%.*f" % (max(1, int(round(-log10(lat_err)))) - 1, lat)
        lons = "%.*f" % (max(1, int(round(-log10(lon_err)))) - 1, lon)
        if '.' in lats: lats = lats.rstrip('0')
        if '.' in lons: lons = lons.rstrip('0')
        _cache[geohash] = (float(lats), float(lons))
        return float(lats), float(lons)
        # return lat, lon

    def geohash(latitude, longitude, precision=6):
        """Encode a position given in float arguments latitude, longitude to
        a geohash which will have the character count precision.
        """
        from math import log10
        __base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
        __decodemap = { }
        for i in range(len(__base32)):
            __decodemap[__base32[i]] = i
        del i
        lat_interval, lon_interval = (-90.0, 90.0), (-180.0, 180.0)
        geohash = []
        bits = [ 16, 8, 4, 2, 1 ]
        bit = 0
        ch = 0
        even = True
        while len(geohash) < precision:
            if even:
                mid = (lon_interval[0] + lon_interval[1]) / 2
                if longitude > mid:
                    ch |= bits[bit]
                    lon_interval = (mid, lon_interval[1])
                else:
                    lon_interval = (lon_interval[0], mid)
            else:
                mid = (lat_interval[0] + lat_interval[1]) / 2
                if latitude > mid:
                    ch |= bits[bit]
                    lat_interval = (mid, lat_interval[1])
                else:
                    lat_interval = (lat_interval[0], mid)
            even = not even
            if bit < 4:
                bit += 1
            else:
                geohash += __base32[ch]
                bit = 0
                ch = 0
        return ''.join(geohash)

    def distance(a,b):
        """Get the distance between to geohashes"""
        return math.sqrt(distance2(a,b))

    def distance2(a,b):
        """Get the distance squared between two geohashes"""
        x0,y0 = geocode(a)
        x1,y1 = geocode(b)
        dx,dy = x0-x1,y0-y1
        return dx*dx+dy*dy

    def nearest(hash,hashlist,withdist=False):
        """Find the nearest geohash in a list of geohashes"""
        if len(hashlist) > 0:
            dist = sorted([(x,distance2(hash,x)) for x in hashlist],key=lambda y:y[1])
            return (dist[0][0],distance(hash,dist[0][0])) if withdist else dist[0][0]
        else:
            return (None,float('nan')) if withdist else None

    #
    # Calendar data
    #
    holidays = []
    def is_workday(date,date_format="%Y-%m-%d %H:%M:%S"):
        global holidays
        if len(holidays) == 0:
            holidays = pd.read_csv("holidays.csv",
                index_col=[0],
                parse_dates=[0],
                date_format="%Y-%m-%d").sort_index()
        if type(date) is str:
            date = dt.datetime.strptime(date,date_format)
        if date.year < holidays.index.min().year or date.year > holidays.index.max().year:
            warning(f"is_workday(date='{date}',date_format='{date_format}') date is not in range of known holidays")
        return date.weekday()<5 and date not in holidays.index

    #
    # Weather data
    #
    def nsrdb_credentials(path=os.path.join(os.environ["HOME"],".nsrdb","credentials.json")):
        try:
            with open(path,"r") as fh:
                return list(json.load(fh).items())[0]
        except Exception as err:
            error(E_INVAL,f"~/.nsrdb/credentials.json read failed - {err}")


    def nsrdb_weather(location,year,
                      interval=30,
                      attributes={"solar[W/m^2]" : "ghi",
                                  "temperature[degC]" : "air_temperature",
                                  "wind[m/s]" : "wind_speed",
                                  'dhi[W/m^2]': 'dhi',
                                  'dni[W/m^2]': 'dni',
                                  'winddirection[deg]': 'wind_direction',
                                  'dewpoint[degC]': 'dew_point',
                                  'relhumidity[pct]': 'relative_humidity',
                                  'water[mm]': 'total_precipitable_water'
                                  }):
        """
        Pull NSRDB data for a particular year and location. 

        Parameters
        ----------
        location: Str.
            Geohash of a particular location that will be decoded to get lat-long
            coordinates.
        year: Int.
            Year of data we want to pull data for.
        interval: Int.
            Frequency of data in minutes. Default 5
        attributes: Dictionary of string keys/values.
            Desired data fields to return as values, and final column names as keys.
            See pvlib documentaton for the full list of fields in NSRDB:
            https://pvlib-python.readthedocs.io/en/v0.9.0/generated/pvlib.iotools.get_psm3.html

        Returns
        -------
        Pandas dataframe containing 'attribute' fields, with UTC ISO format
        datetime index.
        """
        lat,lon = geocode(location)
        leap = (year%4 == 0)
        email, api_key = nsrdb_credentials()
        # Pull from API and save locally
        psm3, _ = pvlib_psm3.get_psm3(lat, lon,
                                      api_key,
                                      email, year,
                                      attributes=attributes.values(),
                                      map_variables=True,
                                      interval=interval,
                                      leap_day=leap,
                                      timeout=60)
        cols_to_remove = ['Year', 'Month', 'Day', 'Hour', 'Minute']
        psm3 = psm3.drop(columns=cols_to_remove)
        psm3.index = pd.to_datetime(psm3.index)
        psm3.rename(columns={"key_0": "datetime",
                             **{v: k for k, v in attributes.items()}},
                    inplace=True)
        psm3 = psm3.round(3)  
        return psm3.sort_index()
    return (
        distance,
        distance2,
        geocode,
        geohash,
        holidays,
        is_workday,
        nearest,
        nsrdb_credentials,
        nsrdb_weather,
    )


if __name__ == "__main__":
    app.run()
