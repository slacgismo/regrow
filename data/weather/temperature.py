import marimo

__generated_with = "0.7.12"
app = marimo.App(width="medium")


@app.cell
def __(mo):
    mo.md(
        r"""
        # REGROW: Temperature Forecasting
         Study of extreme weather and temperature rises in Western Interconnection (WECC) locations.
        """
    )
    return


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import os, sys
    import numpy as np
    import tornado as tn
    from pathlib import Path
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
def __(mo):
    mo.md(r"""## Viewing nodes on Google Earth:""")
    return


@app.cell
def __(Path, __file__, mo):
    # Google Earth Snapshot
    _img = (Path(__file__).parent / 'wecc_google_earth.png')
    mo.image(src=f"{_img}")
    return


@app.cell
def __(geocode, nodes, pd):
    # Manipulating data from nodes to latitude/longitude
    latlong = pd.DataFrame(index=nodes, columns=['lat', 'lon'])

    print(nodes)

    for node in nodes:
        latlong.loc[node] = geocode(node)
    return latlong, node


@app.cell
def __(latlong):
    latlong
    return


@app.cell
def __(temperature):
    nodes = temperature.columns.tolist()
    return nodes,


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
def __(mo):
    mo.md(
        r"""
        ## Comparing Years:

        Analyzing the data from the month of August in years 2018-2022, the peaks in the line graph will displays rise of temperatures from the heatwave.
        """
    )
    return


@app.cell
def __(get_location, temperature):
    # Adjusting timezone
    location = temperature[get_location()]
    return location,


@app.cell
def __(location, pd):
    # Time slicing for August
    august1 = location.loc['2018-08-01':'2018-08-31']
    august2 = location.loc['2019-08-01':'2019-08-31']
    august3 = location.loc['2020-08-01':'2020-08-31']
    august4 = location.loc['2021-08-01':'2021-08-31']

    august1 = pd.DataFrame(august1)
    august2 = pd.DataFrame(august2)
    august3 = pd.DataFrame(august3)
    august4 = pd.DataFrame(august4)
    return august1, august2, august3, august4


@app.cell
def __(august1, august2, august3, august4, mo, plt):
    # Calculated average daily temperatures
    avg_daily_2018 = august1.resample(rule="1D").mean()
    avg_daily_2019 = august2.resample(rule="1D").mean()
    avg_daily_2020 = august3.resample(rule="1D").mean()
    avg_daily_2021 = august4.resample(rule="1D").mean()

    # Plotting the data
    plt.figure(figsize=(9, 5))
    plt.plot(avg_daily_2018.values, label='2018')
    plt.plot(avg_daily_2019.values, label='2019')
    plt.plot(avg_daily_2020.values, label='2020', ls=":")
    plt.plot(avg_daily_2021.values, label='2021')

    plt.xlabel('Date')
    plt.ylabel('Average Temperature (C˚)')
    plt.title('Daily Average Temperature (August 2018-2022)')
    plt.legend()
    plt.gcf().autofmt_xdate() 
    mo.mpl.interactive(plt.gcf())
    return avg_daily_2018, avg_daily_2019, avg_daily_2020, avg_daily_2021


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
