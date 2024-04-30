"""Project load model to other years

"""

import os, sys
import datetime as dt
import pandas as pd
import config
import states
from utils import *

YEAR = 2020
ROUND = 1

if __name__ == "__main__":

	state_list = [states.state_codes_byname[x]["usps"] for x in config.state_list if x in [y[0] for y in states.state_codes]]
	# print(state_list)

	counties = pd.read_csv("counties.csv",index_col=["geocode"])
	counties = counties[counties["usps"].isin(state_list)]

	#
	# Load network data
	#

	network = pd.read_csv("wecc240_gis.csv",usecols=["Bus  Name","Lat","Long"])
	network["geocode"] = [geohash(x,y) for x,y in network[["Lat","Long"]].values]
	network.set_index("geocode",inplace=True)
	network = network[~network.index.duplicated(keep='first')]
	network["county"] = [nearest(x,counties.index.values) for x in network.index.values]
	network["state"] = [counties["usps"][x] for x in network["county"]]
	network["tzoffset"] = [states.state_codes_byusps[x]["tz"] for x in network["state"]]

	#
	# Load source data
	#
	source = {}
	for file in os.listdir("geodata"):
		
		if not file.endswith(".csv"):
			continue

		name = os.path.splitext(file)[0]
		verbose(f"Loading geodata {name}",end="...")
		try:
			source[name] = pd.read_csv(os.path.join("geodata",file),
				index_col=[0],
				parse_dates=[0],
				date_format = "%Y-%m-%d %H:%M:%S+00:00"
				)
			verbose("ok")
		except Exception as err:
			verbose(err)

	#
	# Load target weather
	#
	weather = {}
	for file in os.listdir("weather"):

		if not file.endswith(".csv"):
			continue

		name = os.path.splitext(file)[0]
		verbose(f"Loading weather {name}",end="...")
		try:
			weather[name] = pd.read_csv(os.path.join("weather",file),
				index_col=[0],
				parse_dates=[0],
				date_format = "%Y-%m-%d %H:%M:%S+00:00"
				)
			weather[name] = weather[name][weather[name].index.year==YEAR]
			verbose("ok")
		except Exception as err:
			verbose(err)

	#
	# Load weather sensitivity
	#
	sensitivity = pd.read_csv("sensitivity.csv",index_col=[0]).T

	#
	# Project target data
	#
	target = source["baseload"].copy()
	fromindex = weather["temperature"].index.values
	tolen = len(target.index)
	fromlen = len(fromindex)
	if tolen == fromlen: # exact copy
		target.index = fromindex
	elif tolen < fromlen: # clip
		target.index = fromindex[:tolen]
	elif fromlen > tolen: # repeat last day
		target.index[:fromlen] = fromindex
		target.index[fromlen:] = fromindex[fromlen:] + dt.timedelta(days=1)

	heating = (weather["temperature"][weather["temperature"]<config.Theat]-config.Theat).fillna(0)
	cooling = (weather["temperature"][weather["temperature"]>config.Tcool]-config.Tcool).fillna(0)
	target += heating * sensitivity.loc["heating[MW/degC]"] \
		+ cooling * sensitivity.loc["cooling[MW/degC]"] \
		+ weather["solar"] * sensitivity.loc["solar[MW/W/m^2]"]

	target.round(ROUND).to_csv(f"load_{YEAR}.csv",index=True,header=True)


