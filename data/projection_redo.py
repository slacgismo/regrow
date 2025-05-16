"""Redone projections with new load models"""

import pandas as pd
import load_models as lm
import counties

lm.VERBOSE=False
lm.WARNING=False

# collect county eather and load data
wecc = {x:lm.County(x) for x in counties.data.set_index("region").loc["WECC","geocode"].values}
weather = {}
loads = {}
for geocode,county in wecc.items():
    
    print("Processing",county,end="...",flush=True)

    weather[geocode] = lm.Weather(county)
    loads[geocode] = lm.Loads(county)

    print("ok")

# project loads to target years
years = list(range(2018,2023))
for year in years:

    # get weather for years

    # predict loads

    pass

# aggregate to nodes
nodes = list(pd.read_csv("geodata/temperature.csv",index_col="timestamp").columns)
data = {}
for node in nodes:
    data[node] = []
    for county in counties.find_counties(node,counties,list(wecc),nodes):
        print(county,"-->",node)
        data[node].append(loads)
    data[node] = pd.concat(data[node])