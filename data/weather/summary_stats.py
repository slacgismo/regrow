import marimo

__generated_with = "0.6.26"
app = marimo.App(width="medium")


@app.cell
def __(mo):
    mo.md(
        """
        # REGROW: Magnitude of Heat Waves
        Study of extreme weather and temperature rises in Western Interconnection (WECC) locations by measuring the maginitude of temperature deviation over four years.
        """
    )
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
    return Path, mdates, mo, np, os, pd, plt, sys, tn


@app.cell
def __(Path, __file__, pd):
    # Loading the Data
    _fp = Path(__file__).parent / 'temperature.csv'
    temperature = pd.read_csv(_fp, index_col=0, parse_dates=[0])
    return temperature,


@app.cell
def __(geocode, nodes, pd):
    # Manipulating data from nodes to latitude/longitude
    latlong = pd.DataFrame(index=nodes, columns=['lat', 'lon'])
    for node in nodes:
        latlong.loc[node] = geocode(node)
    return latlong, node


@app.cell
def __(temperature):
    nodes = temperature.columns.tolist()
    return nodes,


@app.cell
def __(mo):
    mo.md(
        """
        ## Case Study of Temperature Deviation:

        Analyzing temperature data in Southern California over several years to compare typical August temperature fluctuations with extreme weather events by identifying patterns and deviations.
        """
    )
    return


@app.cell
def __(Path, __file__, mo):
    # Google Earth Snapshot
    _img = (Path(__file__).parent / 'case_study1.png')
    mo.image(src=f"{_img}")
    return


@app.cell
def __(pd, temperature):
    # Case study of location in SoCal
    location1 = temperature[['9qj152']]
    location1.index = location1.index - pd.Timedelta(7, 'hr')

    # Filtering data for August
    df_august_2018 = location1.loc['2018-08-01':'2018-08-31']
    df_august_2019 = location1.loc['2019-08-01':'2019-08-31']
    df_august_2020 = location1.loc['2020-08-01':'2020-08-31']
    df_august_2021 = location1.loc['2021-08-01':'2021-08-31']

    # Monthly Data Frames 
    df_august_2018 = pd.DataFrame(df_august_2018)
    df_august_2019 = pd.DataFrame(df_august_2019)
    df_august_2020 = pd.DataFrame(df_august_2020)
    df_august_2021 = pd.DataFrame(df_august_2021)

    # Adding 'hour' column for grouping
    df_august_2018["hour"] = df_august_2018.index.hour
    df_august_2019["hour"] = df_august_2019.index.hour
    df_august_2020["hour"] = df_august_2020.index.hour
    df_august_2021["hour"] = df_august_2021.index.hour
    return (
        df_august_2018,
        df_august_2019,
        df_august_2020,
        df_august_2021,
        location1,
    )


@app.cell
def __(mo):
    mo.md("## Daily Standard Deviation")
    return


@app.cell
def __(
    df_august_2018,
    df_august_2019,
    df_august_2020,
    df_august_2021,
    mo,
    np,
    plt,
):
    # Calculated daily standard deviation for each year
    std_daily_2018 = df_august_2018.resample(rule="1D").std()
    std_daily_2019 = df_august_2019.resample(rule="1D").std()
    std_daily_2020 = df_august_2020.resample(rule="1D").std()
    std_daily_2021 = df_august_2021.resample(rule="1D").std()

    # Creating baseline temperature deviation with mean standard deviation of non-heat-wave years
    baseline_daily_temp = (std_daily_2018["9qj152"].values + std_daily_2019["9qj152"].values + std_daily_2021["9qj152"].values)/3

    # Residual (2020 - baseline)
    residual_daily = std_daily_2020["9qj152"].values - baseline_daily_temp

    std_daily_2018["day"] = std_daily_2018.index.day
    std_daily_2019["day"] = std_daily_2019.index.day
    std_daily_2020["day"] = std_daily_2020.index.day
    std_daily_2021["day"] = std_daily_2021.index.day

    # Overall standard deviation of daily average temperatures for each year
    [np.std(std_daily_2018["9qj152"].values), 
     np.std(std_daily_2019["9qj152"].values), 
     np.std(std_daily_2020["9qj152"].values), 
     np.std(std_daily_2021["9qj152"].values)]

    # Plotting Daily Standard Deviation to compare temperature fluctuations 
    plt.figure(figsize=(10, 5))
    plt.plot(std_daily_2018.index.day, baseline_daily_temp, label="baseline")
    plt.plot(std_daily_2018.index.day, std_daily_2020["9qj152"], label="heatwave")
    plt.xlabel('Days in August')
    plt.ylabel('Standard Temp Deviation (°C)')
    plt.title('Daily Temperature Standard Deviation')
    plt.legend()
    mo.mpl.interactive(plt.gcf())
    return (
        baseline_daily_temp,
        residual_daily,
        std_daily_2018,
        std_daily_2019,
        std_daily_2020,
        std_daily_2021,
    )


@app.cell
def __(residual_daily):
    residual_daily.mean()
    return


@app.cell
def __(mo):
    mo.md("## Hourly Standard Deviation")
    return


@app.cell
def __(
    df_august_2018,
    df_august_2019,
    df_august_2020,
    df_august_2021,
    mo,
    plt,
):
    # Hourly standard deviation: Averaging a specific hour from each date of the month (e.g. all 1ams)
    hourly_std_2018 = df_august_2018.groupby("hour").agg({"9qj152":"std"})
    hourly_std_2019 = df_august_2019.groupby("hour").agg({"9qj152":"std"})
    hourly_std_2020 = df_august_2020.groupby("hour").agg({"9qj152":"std"})
    hourly_std_2021 = df_august_2021.groupby("hour").agg({"9qj152":"std"})

    # Creating a baseline by averaging the hourly standard deviations of non-heatwave years
    baseline_hourly_std = (hourly_std_2018 + hourly_std_2019 + hourly_std_2021) / 3

    # Calculating the residuals (2020 hourly std - baseline hourly std)
    hourly_residuals = hourly_std_2020 - baseline_hourly_std

    # Plotting Hourly Standard Deviation
    plt.figure(figsize=(10, 5))
    plt.plot(hourly_std_2018.index, baseline_hourly_std, label='Baseline (2018, 2019, 2021)')
    plt.plot(hourly_std_2018.index, hourly_std_2020, label='2020')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Temperature Standard Deviation (°C)')
    plt.title('Hourly Temperature Standard Deviation')
    plt.legend() 
    mo.mpl.interactive(plt.gcf())
    return (
        baseline_hourly_std,
        hourly_residuals,
        hourly_std_2018,
        hourly_std_2019,
        hourly_std_2020,
        hourly_std_2021,
    )


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
