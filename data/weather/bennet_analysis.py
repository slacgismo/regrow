import marimo

__generated_with = "0.8.11"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    import matplotlib.colors as colors
    import seaborn as sns
    import geopandas
    import geodatasets
    from geodatasets import get_path
    from pathlib import Path
    import os
    from tqdm import tqdm
    return (
        Path,
        TwoSlopeNorm,
        colors,
        geodatasets,
        geopandas,
        get_path,
        mo,
        np,
        os,
        pd,
        plt,
        sns,
        tqdm,
    )


@app.cell
def __(Path, __file__, pd):
    # Loading the Data
    _fp = Path(__file__).parent / "solar.csv"
    solar = pd.read_csv(_fp, index_col=0, parse_dates=[0])

    _fp = Path(__file__).parent / "temperature.csv"
    temperature = pd.read_csv(_fp, index_col=0, parse_dates=[0])

    _fp = Path(__file__).parent / "wind.csv"
    wind = pd.read_csv(_fp, index_col=0, parse_dates=[0])

    # Adjusting for time zone differences (shifting by 8 hours)
    solar.index = solar.index - pd.Timedelta(8, "hr")
    temperature.index = temperature.index - pd.Timedelta(8, "hr")
    return solar, temperature, wind


@app.cell
def __(nearest, os, pd, wind):
    nodes = wind.columns.tolist()
    counties = pd.read_csv(os.path.join("..", "counties.csv"), index_col="geocode")
    nodes_counties = dict(
        [(x, counties.loc[nearest(x, counties.index)].county) for x in nodes]
    )
    return counties, nodes, nodes_counties


@app.cell
def __(analyze, mo, nodes, nodes_counties, temperature):
    _n = nodes[13]
    mo.hstack([_n, nodes_counties[_n], analyze(temperature, _n)])
    return


@app.cell
def __(analyze, nodes, pd, solar, temperature, tqdm, wind):
    results = pd.DataFrame(
        index=nodes, columns=["temp_anomoly", "solar_anomoly", "wind_anomoly"]
    )
    for _n in tqdm(nodes):
        _record = []
        for _ds in [temperature, solar, wind]:
            _record.append(analyze(_ds, _n))
        results.loc[_n] = _record
    return results,


@app.cell
def __(mo, results):
    mo.ui.table(results)
    return


@app.cell
def __(plt, results, sns):
    sns.jointplot(data=results, x="temp_anomoly", y="solar_anomoly")
    plt.gcf()
    return


@app.cell
def __(geocode, nodes, pd, results):
    # Manipulating data from nodes to latitude/longitude
    latlong = pd.DataFrame(index=nodes, columns=["lat", "lon"])
    for geo_node in nodes:
        latlong.loc[geo_node] = geocode(geo_node)
    latlong = latlong.join(results)
    return geo_node, latlong


@app.cell
def __(colors, latlong, sns):
    sns.relplot(
        x="lon",
        y="lat",
        hue="temp_anomoly",
        palette="coolwarm",
        data=latlong,
        norm=colors.CenteredNorm(),
    )
    return


@app.cell
def __(geopandas, latlong):
    gdf = geopandas.GeoDataFrame(
        latlong,
        geometry=geopandas.points_from_xy(latlong.lon, latlong.lat),
        crs="EPSG: 4326",
    )
    return gdf,


@app.cell
def __(gdf):
    gdf
    return


@app.cell
def __(geopandas, get_path, latlong, plt, sns):
    def make_geoplot(column):
        world = geopandas.read_file(get_path("naturalearth.land"))
        g = sns.relplot(
            x="lon", y="lat", hue=column, palette="coolwarm", data=latlong
        )
        g._legend.remove()
        _ax = plt.gca()
        world.clip([-130, 30, -102, 51]).plot(
            color="none", edgecolor="black", ax=_ax
        )
        norm = plt.Normalize(latlong[column].min(), latlong[column].max())
        sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
        # _ax.get_legend().remove()
        _ax.figure.colorbar(
            sm, ax=_ax, fraction=0.046, pad=0.04, label="% change from baseline"
        )
        g.fig.suptitle(column)
        return plt.gcf()
    return make_geoplot,


@app.cell
def __(make_geoplot, mo, plt):
    make_geoplot("temp_anomoly")
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __(make_geoplot, mo, plt):
    make_geoplot("solar_anomoly")
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __(make_geoplot, mo, plt):
    make_geoplot("wind_anomoly")
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def __(latlong):
    latlong
    return


@app.cell
def __(np):
    def analyze(data, node):
        august1 = data.loc["2018-08-14":"2018-08-15", node]
        august2 = data.loc["2019-08-14":"2019-08-15", node]
        august3 = data.loc["2020-08-14":"2020-08-15", node]
        august4 = data.loc["2021-08-14":"2021-08-15", node]
        august5 = data.loc["2022-08-15":"2022-08-15", node]

        baseline_avg = np.average(np.r_[august1, august2, august4, august5])
        event_avg = np.average(august3)

        return 100 * (event_avg - baseline_avg) / baseline_avg
    return analyze,


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
        __base32 = "0123456789bcdefghjkmnpqrstuvwxyz"
        __decodemap = {}
        for i in range(len(__base32)):
            __decodemap[__base32[i]] = i
        del i
        lat_interval, lon_interval = (-90.0, 90.0), (-180.0, 180.0)
        lat_err, lon_err = 90.0, 180.0
        is_even = True
        for c in geohash:
            cd = __decodemap[c]
            for mask in [16, 8, 4, 2, 1]:
                if is_even:  # adds longitude info
                    lon_err /= 2
                    if cd & mask:
                        lon_interval = (
                            (lon_interval[0] + lon_interval[1]) / 2,
                            lon_interval[1],
                        )
                    else:
                        lon_interval = (
                            lon_interval[0],
                            (lon_interval[0] + lon_interval[1]) / 2,
                        )
                else:  # adds latitude info
                    lat_err /= 2
                    if cd & mask:
                        lat_interval = (
                            (lat_interval[0] + lat_interval[1]) / 2,
                            lat_interval[1],
                        )
                    else:
                        lat_interval = (
                            lat_interval[0],
                            (lat_interval[0] + lat_interval[1]) / 2,
                        )
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
            return _cache[geohash][0], _cache[geohash][1]
        lati, long, lat_err, lon_err = _decode(geohash)
        from math import log10

        # Format to the number of decimals that are known
        lats = "%.*f" % (max(1, int(round(-log10(lat_err)))) - 1, lati)
        lons = "%.*f" % (max(1, int(round(-log10(lon_err)))) - 1, long)
        if "." in lats:
            lats = lats.rstrip("0")
        if "." in lons:
            lons = lons.rstrip("0")
        _cache[geohash] = (float(lati), float(long))
        return float(lati), float(long)
        # return lat, lon


    def geohash(latitude, longitude, precision=6):
        """Encode a position given in float arguments latitude, longitude to
        a geohash which will have the character count precision.
        """
        from math import log10

        __base32 = "0123456789bcdefghjkmnpqrstuvwxyz"
        __decodemap = {}
        for i in range(len(__base32)):
            __decodemap[__base32[i]] = i
        del i
        lat_interval, lon_interval = (-90.0, 90.0), (-180.0, 180.0)
        geohash = []
        bits = [16, 8, 4, 2, 1]
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
        return "".join(geohash)


    def distance(a, b):
        """Get the distance between to geohashes"""
        return math.sqrt(distance2(a, b))


    def distance2(a, b):
        """Get the distance squared between two geohashes"""
        x0, y0 = geocode(a)
        x1, y1 = geocode(b)
        dx, dy = x0 - x1, y0 - y1
        return dx * dx + dy * dy


    def nearest(hash, hashlist, withdist=False):
        """Find the nearest geohash in a list of geohashes"""
        if len(hashlist) > 0:
            dist = sorted(
                [(x, distance2(hash, x)) for x in hashlist], key=lambda y: y[1]
            )
            return (
                (dist[0][0], distance(hash, dist[0][0]))
                if withdist
                else dist[0][0]
            )
        else:
            return (None, float("nan")) if withdist else None


    #
    # Calendar data
    #
    holidays = []


    def is_workday(date, date_format="%Y-%m-%d %H:%M:%S"):
        global holidays
        if len(holidays) == 0:
            holidays = pd.read_csv(
                "holidays.csv",
                index_col=[0],
                parse_dates=[0],
                date_format="%Y-%m-%d",
            ).sort_index()
        if type(date) is str:
            date = dt.datetime.strptime(date, date_format)
        if (
            date.year < holidays.index.min().year
            or date.year > holidays.index.max().year
        ):
            warning(
                f"is_workday(date='{date}',date_format='{date_format}') date is not in range of known holidays"
            )
        return date.weekday() < 5 and date not in holidays.index


    #
    # Weather data
    #
    def nsrdb_credentials(
        path=os.path.join(os.environ["HOME"], ".nsrdb", "credentials.json"),
    ):
        try:
            with open(path, "r") as fh:
                return list(json.load(fh).items())[0]
        except Exception as err:
            error(E_INVAL, f"~/.nsrdb/credentials.json read failed - {err}")


    def nsrdb_weather(
        location,
        year,
        interval=30,
        attributes={
            "solar[W/m^2]": "ghi",
            "temperature[degC]": "air_temperature",
            "wind[m/s]": "wind_speed",
            "dhi[W/m^2]": "dhi",
            "dni[W/m^2]": "dni",
            "winddirection[deg]": "wind_direction",
            "dewpoint[degC]": "dew_point",
            "relhumidity[pct]": "relative_humidity",
            "water[mm]": "total_precipitable_water",
        },
    ):
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
        lat, lon = geocode(location)
        leap = year % 4 == 0
        email, api_key = nsrdb_credentials()
        # Pull from API and save locally
        psm3, _ = pvlib_psm3.get_psm3(
            lat,
            lon,
            api_key,
            email,
            year,
            attributes=attributes.values(),
            map_variables=True,
            interval=interval,
            leap_day=leap,
            timeout=60,
        )
        cols_to_remove = ["Year", "Month", "Day", "Hour", "Minute"]
        psm3 = psm3.drop(columns=cols_to_remove)
        psm3.index = pd.to_datetime(psm3.index)
        psm3.rename(
            columns={"key_0": "datetime", **{v: k for k, v in attributes.items()}},
            inplace=True,
        )
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
