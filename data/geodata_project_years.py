"""Project load data to other years

Recipe
------

1. Import load data from 2018 baseline (extracted from NREL ResStock and
   ComStock)

2. Change from lagging to leading time index if necessary.

3. Project load data from 2018 into 2019-2021, adjusting for DOW

4. Compute how much weather changes from 2018 to 2019-2021.

5. Adjust loads according the weather change and heat/cool sensitivity.

6. Compute total load

"""

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import utils
import config

FROM_YEAR = 2018
TO_YEARS = [2019,2020,2021]

WEATHER = ["temperature","solar","wind"]
LOADS = ["baseload","cooling","heating","total"]
UNITS = {"temperature":"$^o$C", "solar":"W/M$^2$", "wind":"m/s"}
DATASETS = WEATHER + LOADS
FILES = dict((x,f"geodata/{x}_{FROM_YEAR}.csv") for x in DATASETS)

PLOT = False # True to enable plot output (slower)
FIGSIZE=(20,10)

#
# 1. Import load data
#
DATA = dict((x,pd.read_csv(FILES[x],index_col=[0],parse_dates=[0])) 
	for x in FILES.keys())

#
# 2. Fix datetime index to be all leading
#
for data in DATA.values():
	shift = -1 if data.index[0].to_datetime64() == dt.datetime(FROM_YEAR,1,1,1,0,0) else 0
	data.index = (data.index.tz_localize('UTC') + pd.Timedelta(hours=shift)) 

#
# 3. Project loads into years, adjusting for DOW
#
def get_offset_shift(year):
	OFFSET = int((dt.datetime(year=year,month=1,day=1) - 
		dt.datetime(year=FROM_YEAR,month=1,day=1)).total_seconds()/3600)
	SHIFT = int(OFFSET%(24*7))
	return OFFSET,SHIFT

print("Projecting dataset",flush=True,end="")
for dataset in LOADS:
	data = [DATA[dataset]]
	for year in TO_YEARS:

		OFFSET,SHIFT = get_offset_shift(year) 

		data.append(pd.concat([
			DATA[dataset].iloc[SHIFT:],
			DATA[dataset].iloc[:SHIFT],
			]))
		data[-1].index = DATA[dataset].index + pd.Timedelta(hours=OFFSET)
		print(".",end="",flush=True)
	DATA[dataset] = pd.concat(data)	
print("OK")

#
# 4. Mapping weather data to node (matching columns)
#
print("Mapping weather",flush=True,end="...")
for dataset in WEATHER:
	data = []
	source = pd.read_csv(f"weather/{dataset}.csv",index_col=[0],parse_dates=[0])
	for column in DATA[dataset]:
		nearest = utils.nearest(column,source.columns)
		data.append(pd.DataFrame(source[column]))
	DATA[dataset] = pd.concat(data,axis=1)
	if PLOT:
		print("plotting",dataset,end="...",flush=True)
		plt.figure()
		DATA[dataset].plot(figsize=FIGSIZE,
			xlabel="Date/Time (UTC)",
			ylabel=f"{dataset.title()} ({UNITS[dataset]})",
			legend=False,
			grid=True,
			)
		plt.savefig(f"geodata/{dataset}.png")
TEMP = [DATA["temperature"]]
print("OK")

#
# 4. Compute weather change
#
print("Computing weather changes",flush=True,end="")
TEMP = [DATA["temperature"][:8760]] # start with reference temperatures
for year in TO_YEARS:
	print(".",end="",flush=True)
	OFFSET,SHIFT = get_offset_shift(year) 

	TEMP.append(pd.concat([TEMP[0].iloc[SHIFT:],TEMP[0].iloc[:SHIFT]]))
	TEMP[-1].index = TEMP[0].index + pd.Timedelta(hours=OFFSET)
TEMP = pd.concat(TEMP) # time adjusted reference temperatures to match loads
DT = TEMP - DATA["temperature"]	# temperature differences
if PLOT:
	print("plotting",end="",flush=True)
	for dataset in ["cooling","heating"]:
		plt.figure(figsize=FIGSIZE)
		plt.scatter(x=DATA["temperature"][:8760],y=DATA[dataset][:8760],marker='.')
		plt.grid()
		plt.xlabel("Temperature ($^o$C)")
		plt.ylabel(f"{dataset.title()} Load (MW)")
		plt.savefig(f"geodata/{dataset}_temperature.png")
		print(".",end="",flush=True)
print("OK")

#
# 5. Adjust heating/cooling loads
#
print("Adjusting loads",flush=True,end="")
temperature_sensitivity = pd.read_csv("sensitivity.csv",index_col=[0])
for dataset in ["heating","cooling"]:
	print(".",end="",flush=True)
	data = DATA[dataset]
	dload = DT.copy()*temperature_sensitivity[f"{dataset}[MW/degC]"].transpose()
	DATA[dataset] = (data + dload).clip(lower=0).dropna().round(4)
print("OK")

#
# 6. Compute total load
#
print("Updating total load",flush=True,end="...")
DATA["total"] = (DATA["baseload"] + DATA["heating"] + DATA["cooling"]).dropna().round(4)
print("OK")

#
# Final plots
#
if PLOT:
	print("Generating plots",flush=True,end="")
	for dataset in LOADS:
		plt.figure()
		DATA[dataset].plot(figsize=FIGSIZE,
			xlabel="Date/Time (UTC)",
			ylabel=dataset.title()+" (MW)",
			legend=False,
			grid=True,
			)
		plt.savefig(f"geodata/{dataset}.png")
		print(".",end="",flush=True)
	print("OK")

#
# Save datasets
#
print("Saving datasets",flush=True,end="")
for name,data in DATA.items():
	data.to_csv(f"geodata/{name}.csv",index=True,header=True)
	print(".",end="",flush=True)
print("OK")

