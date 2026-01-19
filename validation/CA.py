import sys
import os
import json
import pandas as pd
import io
import requests
import datetime as dt
import pytz

with open(os.path.join(os.environ["HOME"],".eia","api_key-2"),"r") as fh:
    apikey = fh.read()

year = 2020
for month in range(8,9):
    start = f"{year}-{month:02d}"
    end = f"{year+1 if month == 12 else year}-{month+1 if month < 12 else month:02d}"
    fmt = "%Y-%m-%dT%H"
    url = f"https://api.eia.gov/v2/electricity/rto/region-data/data/?frequency=hourly&data[0]=value&facets[respondent][]=CAL&facets[type][]=D&facets[type][]=NG&facets[type][]=TI&start={start}-01T00&end={end}-01T00&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000&api_key={apikey}"
    # url = f"https://api.eia.gov/v2/electricity/rto/region-data/data/?frequency=hourly&data[0]=value&facets[respondent][]=CAL&facets[type][]=D&facets[type][]=NG&facets[type][]=TI&start=2020-08-01T00&end=2020-08-31T00&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000&api_key=GEgWg1lHgGyH1h7WbmqCPfJgGkquSbnY79e2Aonn"
    # url = f"https://api.eia.gov/v2/electricity/rto/region-data/data/?frequency=hourly&data[0]=value&facets[respondent][]=CAL&facets[type][]=D&facets[type][]=NG&facets[type][]=TI&start=2020-08-01T00&end=2020-08-31T23&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000&api_key=GEgWg1lHgGyH1h7WbmqCPfJgGkquSbnY79e2Aonn"
    url = "https://api.eia.gov/v2/electricity/rto/region-data/data/?frequency=hourly&data[0]=value&facets[respondent][]=CAL&facets[type][]=D&facets[type][]=NG&facets[type][]=TI&start=2020-08-01T00&end=2020-09-01T00&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000&api_key=GEgWg1lHgGyH1h7WbmqCPfJgGkquSbnY79e2Aonn"
    req = requests.get(url)
    if req.status_code != 200:
        raise RuntimeError(f"{url} --> HTTP {req.status_code}")
    data = pd.DataFrame(json.loads(req.text)["response"]["data"])
    data.period = pd.DatetimeIndex(data.period)
    data.set_index(["respondent","period","type"],inplace=True)
    data.drop(["respondent-name","type-name","value-units"],inplace=True,axis=1)
    data = data.unstack().sort_index().iloc[:-1].reset_index()
    data.columns = ["operator","timestamp","load","generation","imports"]

    print(data)