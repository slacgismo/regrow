import os, sys
import datetime as dt
import pandas as pd
import config
import states
from utils import *

if __name__ == "__main__":

	state_list = [states.state_codes_byname[x]["usps"] for x in config.state_list if x in [y[0] for y in states.state_codes]]
	# print(state_list)

	counties = pd.read_csv("counties.csv",index_col=["geocode"])
	counties = counties[counties["usps"].isin(state_list)]

	network = pd.read_csv("wecc240_gis.csv",usecols=["Bus  Name","Lat","Long"])
	network["geocode"] = [geohash(x,y) for x,y in network[["Lat","Long"]].values]
	network.set_index("geocode",inplace=True)
	network = network[~network.index.duplicated(keep='first')]
	network["county"] = [nearest(x,counties.index.values) for x in network.index.values]
	network["state"] = [counties["usps"][x] for x in network["county"]]
	network["tzoffset"] = [states.state_codes_byusps[x]["tz"] for x in network["state"]]

	network.sort_index().to_csv("nodes.csv",index=True,header=True)