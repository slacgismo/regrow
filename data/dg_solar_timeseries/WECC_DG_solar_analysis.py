# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "geopandas==1.1.3",
#     "marimo>=0.23.8",
#     "numpy==2.4.6",
#     "pandas==3.0.3",
#     "requests==2.34.2",
#     "shapely==2.1.2",
# ]
# ///

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")


@app.cell
def _():
    from __future__ import annotations

    import marimo as mo
    import pandas as pd
    import numpy as np

    import requests
    import geopandas as gpd
    from shapely.geometry import Point
    import json
    import time

    return Point, gpd, np, pd


@app.cell
def _():
    return


@app.cell
def _():
    short_to_long = {
        'NM': 'New Mexico', 
        'CA': 'California', 
        'WY': 'Wyoming', 
        'OR': 'Oregon', 
        'UT': 'Utah', 
        'WA': 'Washington', 
        'AZ': 'Arizona', 
        'NV': 'Nevada', 
        'ID': 'Idaho', 
        'MT': 'Montana', 
        'CO': 'Colorado'
    }
    long_to_short = {value: key for key, value in short_to_long.items()}
    return (long_to_short,)


@app.cell
def _(Point, gpd):
    # Load US states shapefile. This is a sample URL; you can find others or download and point to your local file.
    STATES_URL = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/refs/heads/master/data/geojson/us-states.json"

    # Load GeoDataFrame with US states
    states = gpd.read_file(STATES_URL)

    def get_location(lat, lon):
        point = Point(lon, lat)  # Note: (lon, lat) order is used for Point
        state_found = states[states['geometry'].contains(point)]

        if not state_found.empty:
            # If a state is found, return the name of the state
            return state_found.iloc[0]['name']  # Adjust this key based on the GeoJSON properties
        else:
            # If no state is found, call an API to get the country
            return get_country(lat, lon)

    def get_country(lat, lon):
        # Using a free API for country lookup
        # response = requests.get(f'http://geocode.xyz/{lat},{lon}?json=1')
        # if response.status_code == 200:
        #     data = response.content
        #     response_json = json.loads(response.content.decode('utf-8'))
        #     provence = response_json['prov']
        #     if provence == 'Throttled! See geocode.xyz/pricing':
        #         # oops, wait a second for free API to unlock...
        #         time.sleep(1)
        #         response = requests.get(f'http://geocode.xyz/{lat},{lon}?json=1')
        #         response_json = json.loads(response.content.decode('utf-8'))
        #         provence = response_json['prov']
        #     if provence == 'MX':
        #         country = 'Mexico'
        #     elif provence == 'CA':
        #         country = 'Canada'
        #     elif provence == 'US':
        #         # If in the US, return the state
        #         country = response_json['statename']
        #     else:
        #         country = provence
        #     return country
        # else:
        #     return 'Unable to find country'
        if lat < 40:
            return "Mexico"
        else:
            return "Canada"

    return (get_location,)


@app.cell
def _(get_location):
    ## Test State/Country lookup
    # lat = 51.5074  # Latitude for London
    # lon = -0.1278  # Longitude for London
    lat = 47.6321 # Spokane, WA
    lon = -117.478965 # Spokane, WA
    # lat = 12.3548 # Hanoi, Vietnam
    # lon = 108.4654 # Hanoi, Vietnam
    location = get_location(lat, lon)
    print(location)  
    return


@app.cell
def _(pd):
    wecc_dg_data = pd.read_csv('wecc_dg_solar.csv')
    # numbers have commas which causes data to be read as strings instead of floats
    for _col in wecc_dg_data.columns:
        if _col not in ['State', 'Data Status']:
            try:
                wecc_dg_data[_col] = wecc_dg_data[_col].str.replace(',', '').astype(float)
            except AttributeError:
                pass
    return (wecc_dg_data,)


@app.cell
def _(wecc_dg_data):
    wecc_dg_data
    return


@app.cell
def _(get_location):
    get_location(32.227887,-115.436076)
    return


@app.cell
def _(get_location, np, pd):
    wecc_bus_summary = pd.read_csv('test_wecc240_2020m_gis.csv')
    wecc_bus_summary['state'] = [get_location(_row['LAT'], _row['LON']) for _, _row in wecc_bus_summary.iterrows()]
    wecc_bus_summary['load_fraction'] = 0.0
    grouped = wecc_bus_summary[wecc_bus_summary['LOAD'] > 0].groupby('state')
    for _state in set(wecc_bus_summary['state']):
        load_frac = grouped.get_group(_state)['LOAD'] / np.sum(grouped.get_group(_state)['LOAD'])
        wecc_bus_summary.loc[grouped.get_group(_state).index, 'load_fraction'] = load_frac
    return (wecc_bus_summary,)


@app.cell
def _(wecc_bus_summary):
    wecc_bus_summary
    return


@app.cell
def _(long_to_short, pd, wecc_bus_summary, wecc_dg_data):
    dataframe_list = []
    grouped_dg = wecc_dg_data.groupby('State')
    for _ix, _row in wecc_bus_summary.iterrows():
        include = _row['load_fraction'] > 0 and _row['state'] not in ['Canada', 'Mexico']
        if include:
            new_df = pd.DataFrame(columns=['Year', 'Month', 'State', 'bus_id', 'bus_name', 'geohash', 'lat', 'lon', 'Capacity [MW]', 'Generation [MWh]'])
            for _ix2, _row2 in grouped_dg.get_group(long_to_short[_row['state']]).iterrows():
                _e = [_row2['Year'], _row2['Month'], _row2['State'], _row['BUS_I'], _row['NAME'], 
                      _row['GEOHASH'], _row['LAT'], _row['LON'], 
                      float(_row['load_fraction']) * float(_row2['Total Capacity (MW)']), 
                      float(_row['load_fraction']) * float(_row2['Total Generation (MWh)'])]
                new_df.loc[_ix2] = _e
            dataframe_list.append(new_df)
    return (dataframe_list,)


@app.cell
def _(dataframe_list):
    dataframe_list
    return


@app.cell
def _(dataframe_list, np, pd):
    wecc_bus_dg_cap_and_gen_by_month = pd.concat(dataframe_list)
    wecc_bus_dg_cap_and_gen_by_month.index = np.arange(len(wecc_bus_dg_cap_and_gen_by_month))
    return (wecc_bus_dg_cap_and_gen_by_month,)


@app.cell
def _(wecc_bus_dg_cap_and_gen_by_month):
    wecc_bus_dg_cap_and_gen_by_month
    return


@app.cell
def _(wecc_bus_dg_cap_and_gen_by_month):
    wecc_bus_dg_cap_and_gen_by_month.to_csv('wecc_bus_dg_cap_and_gen_by_month.csv')
    return


if __name__ == "__main__":
    app.run()
