import marimo

__generated_with = "0.7.3"
app = marimo.App(width="medium")


@app.cell
def __(mo):
    mo.md(r"# REGROW: Temperature Forecasting")
    return


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import utils
    import os, sys
    import numpy as np
    return mdates, mo, np, os, pd, plt, sns, sys, utils


@app.cell
def __(mo):
    mo.md(r"## Geographic location coding")
    return


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


@app.cell
def __(mo):
    mo.md(r"## Loading the Data")
    return


@app.cell
def __(__file__, pd):
    # Marimo special feature (_varible), cell-specific identified, pulling from local computer
    # _fp = "/Users/melody/Documents/REGROW/regrow/data/weather/temperature.csv"

    from pathlib import Path

    _fp = Path(__file__).parent / 'temperature.csv'

    # Organizing columns 
    temperature = pd.read_csv(_fp, index_col=0, parse_dates=[0])
    return Path, temperature


@app.cell
def __(temperature):
    temperature
    return


@app.cell
def __(mo):
    mo.md(r"## Manipulating the Data")
    return


@app.cell
def __(geocode, nodes, pd):
    latlong = pd.DataFrame(index=nodes, columns=['lat', 'lon'])
    for node in nodes:
        # key = value
        latlong.loc[node] = geocode(node)
            # arrays, dic.   # functions

    latlong
    return latlong, node


@app.cell
def __(temperature):
    # time slicing
    temperature.loc["2018-03":"2018-07-15"]
    return


@app.cell
def __(temperature):
    # Remove column header
    nodes = temperature.columns.tolist()[1:]
    return nodes,


@app.cell
def __(nodes):
    nodes
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Checking nodes convert to location:

        For graphical consistency and accuracy using latitude and longitude data.
        """
    )
    return


@app.cell
def __(geohash):
    # converting from latlong to nodes
    geohash(32.23, -115.4)
    return


@app.cell
def __(geocode, nodes):
    # converting from nodes to latlong
    geocode(nodes[0])
    return


@app.cell
def __(geohash):
    geohash(36.91, -108.28)
    return


@app.cell
def __(geocode, geohash, nodes):
    geohash(*geocode(nodes[0]))
    return


@app.cell
def __(nodes):
    # Picked out one specific node
    code = nodes[0]
    code
    return code,


@app.cell
def __(geocode, geohash, nodes):
    geocode(geohash(*geocode(nodes[0])))
    return


@app.cell
def __(code, geocode):
    # Breaking it down by latitude and longitude
    geocode(code)
    return


@app.cell
def __(code, geocode, geohash):
    # Converting latlong back to a node
    geocode(geohash(*geocode(code)))
    return


@app.cell
def __(mo):
    mo.md(r"## Plotting the Data")
    return


@app.cell
def __(mo, pd, plt, temperature):
    # Setting a varible, picking one node in New Mexico
    data_view = temperature[['9r106e']]
    data_view.index = data_view.index - pd.Timedelta(7, 'hr')

    # Plotting the node
    data_view.plot()

    # Converting static grid to zoom-able view
    mo.mpl.interactive(plt.gcf())

    plt.xlabel('Year')
    plt.ylabel('Average Temperature')
    plt.title('Temperature in New Mexico (2018-2022)')
    return data_view,


@app.cell
def __(mo):
    mo.md(r"## Heat Map")
    return


@app.cell
def __(data_view):
    '''
    Pulling together axes for heat map. 
    24 hours (midnight to day to night), 365 days x 4 years (-1 puts year on x-axis)
    '''
    my_data_array = data_view.loc['2018-01-01':'2021-12-30'].values.reshape((24, -1), order='F')
    my_data_array.shape
    return my_data_array,


@app.cell
def __(my_data_array, plt, sns):
    # Heat map feature! 
    # gcf - get current figure 
    # cmap - color map

    sns.heatmap(my_data_array, cmap="plasma")
    plt.gcf()

    plt.xlabel('Days over 4 Years')
    plt.ylabel('Hours of the Day')
    plt.title('Temperature in New Mexico (2018-2022)')
    return


@app.cell
def __(mo):
    mo.md(r"## Time Slicing: One Month of Data")
    return


@app.cell
def __(mo, pd, plt, temperature):
    # Location near Gilroy, CA 
    gilroy = temperature[['9q97v8']]

    # Adjusting timezone
    gilroy.index = gilroy.index - pd.Timedelta(7, 'hr')

    # Time slices - August 2018 to 2022
    august1 = gilroy.loc['2018-08-01':'2018-08-31']
    august2 = gilroy.loc['2019-08-01':'2019-08-31']
    august3 = gilroy.loc['2020-08-01':'2020-08-31']
    august4 = gilroy.loc['2021-08-01':'2021-08-31']

    # New data frame
    august1 = pd.DataFrame(august1)
    august2 = pd.DataFrame(august2)
    august3 = pd.DataFrame(august3)
    august4 = pd.DataFrame(august4)

    # Graphing
    august1.plot()
    mo.mpl.interactive(plt.gcf())

    plt.xlabel('Date and Time')
    plt.ylabel('Average Temperature') 
    plt.title('Hourly temp in Gilroy (August 2018)')
    return august1, august2, august3, august4, gilroy


@app.cell
def __(mo):
    mo.md(r"## Averaging the data:")
    return


@app.cell
def __(august1, plt):
    # Using portion of Data frame, to take date and time, then average it 
    gilroy_avg = august1.resample(rule="1D").mean()
    gilroy_avg.plot()

    plt.xlabel('Date and Time')
    plt.ylabel('Average Temperature')
    plt.title('Daily Avg Temperature (August 2018)')
    return gilroy_avg,


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Comparing Years - Heat Wave:

        Comparing the years 2018-2022, the visble peak of **2020** displays rise of temperatures from the heatwave.
        """
    )
    return


@app.cell
def __(august1, august2, august3, august4, mo, plt):
    # Plotting all years together
    avg1 = august1.resample(rule="1D").mean()
    avg2 = august2.resample(rule="1D").mean()
    avg3 = august3.resample(rule="1D").mean()
    avg4 = august4.resample(rule="1D").mean()

    # Plotting thy data
    plt.figure(figsize=(12, 6))
    plt.plot(avg1.values, label='2018')
    plt.plot(avg2.values, label='2019')
    plt.plot(avg3.values, label='2020', ls=":")
    plt.plot(avg4.values, label='2021')

    plt.gcf().autofmt_xdate() # adjusts x-axis dates
    mo.mpl.interactive(plt.gcf()) # marimo feature to zoom in and out

    plt.xlabel('Date')
    plt.ylabel('Average Temperature')
    plt.title('Daily Average Temperature (August 2018-2022)')
    plt.legend() # adds info box
    return avg1, avg2, avg3, avg4


@app.cell
def __(mo):
    mo.md(r"### Viewing nodes on a global map: Used Google Earth")
    return


@app.cell
def __():
    # Opened in Google Earth
    # latlong.to_csv('latlon_for_google_earth.csv')
    return


if __name__ == "__main__":
    app.run()
