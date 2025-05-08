"""Create load models for nodes

Procedure:

1. Estimate the hourly total load for each quantile from 2018-2022 based on
2018 load data.

2. Compute which quantile the observed weather is from 2018-2022.

3. Select (or interpolate) the hourly total loads from the quantiles based on
observed weather quantile.

4. Aggregate the selected (or interpolated) hourly total loads to each node.

"""

def verbose(x,end=None):
    print(x,end="" if x.endswith("...") or end==False else "\n",flush=True)

verbose("Loading modules...")
import os
import sys
import datetime as dt
import pandas as pd
import numpy as np
import spcqe as qe
import tzinfo
verbose("ok")

# import tzinfo
# UTC = tzinfo.TZ("UTC",0,0)
# timezones = {x:tzinfo.TZ(x,*y) for x,y in {
#     "EST":(-5,0),
#     "CST":(-6,0),
#     "MST":(-7,0),
#     "PST":(-8,0),
#     }.items()}

QUANTILES = [0.02,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.98]
HOURS = pd.date_range("2018-01-01 00:00:00+00:00","2023-01-01 08:00:00+00:00",freq="1h")

verbose("Reading node total load data...")
total = pd.read_csv("geodata/total_2018.csv",
    index_col=["timestamp"],
    parse_dates=True,
    low_memory=True,
    ).dropna()
total.index = total.index.tz_localize("UTC")
verbose(f" {len(total)} records ok")

verbose("Reading node weather data...")
weather = pd.read_csv("geodata/temperature.csv",
    index_col=["timestamp"],
    parse_dates=True,
    low_memory=True,
    ).dropna()
verbose(f" {len(weather)} records ok")

verbose("Extracting node geocodes...")
geocodes = weather.columns
verbose(f" {len(geocodes)} geocodes ok")

# verbose("Reading node data...")
# counties = pd.read_csv("counties.csv")
# counties.drop(counties[~counties["geocode"].isin(geocodes)].index,inplace=True)
# counties.set_index("geocode",inplace=True)
# verbose(f" {len(counties)} counties ok")

verbose("Computing node-level quantiles... ")

def estimates(samples,quantiles=QUANTILES):
    index = np.array([int((x.timestamp()-HOURS[0].timestamp())/3600) for x in HOURS])
    spq = qe.SmoothPeriodicQuantiles(
        num_harmonics=3,
        periods=[365.24 * 24, 7 * 24, 24],
        weight=0.1,
        quantiles=quantiles
    )
    data = np.full(max(index)+1,float('nan'))
    data[:len(samples[node])] = samples
    spq.fit(data)
    result = pd.DataFrame(
        data=spq.fit_quantiles[index,:],
        index=HOURS,
        columns=[f"q{x * 100:02.0f}" for x in QUANTILES],
        )
    result.index.name = "timestamp"
    return result

def find_quantiles(samples,quantiles):
    data = pd.DataFrame(pd.concat([samples,quantiles],axis=1)).dropna()
    geocode = data.columns[0]
    columns = {f"q{x * 100:02.0f}":x for x in QUANTILES}
    for t,data in data.iterrows():
        if t.hour == 0:
            verbose(f"\r    Finding temperature quantiles ({t})... ")
        x = float(data[geocode])
        cdf = [(columns[x],y) for x,y in data[1:].to_dict().items()]
        print(f"{t=}, {geocode=}, {x=}, {cdf=}")

temperatures = "geodata/temperatures"
os.makedirs(temperatures,exist_ok=True)
for n,node in enumerate(weather.columns):

    verbose(f"  {node} ({n+1}/{len(weather.columns)})... ")
    file = f"{temperatures}/{node}.csv"
    
    if os.path.exists(file):
    
        verbose("    Reloading temperature quantiles... ")
        temperature = pd.read_csv(file,index_col=["timestamp"],parse_dates=["timestamp"])
    
    else:
    
        verbose("    Estimating temperature quantiles... ")
        temperature = estimates(weather[node])
        temperature.round(3).to_csv(file,header=True,index=True)

    weather_quantiles = find_quantiles(weather[name],temperature)
    print(weather_quantiles)

    # verbose("Estimating power quantiles... ")
    # power = estimates(total[node.geocode])
    # print(power)

    break

    verbose("ok",True)



