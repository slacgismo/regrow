"""Update weather geodata files"""

import os, sys
import pandas as pd

outputs = {
	"Unnamed: 0" : None,
	"ghi" : "solar",
	"temp_air" : "temperature",
	"wind_speed" : "wind",
	}

resample = "1h"
precision = 1
output_folder = "geodata"
compression = None

results = {}
files = os.listdir("nsrdb")
for n,file in enumerate(files):
	if not file.endswith(".csv.zip"):
		continue
	geocode = file.replace(".csv.zip","")
	print("Processing",file,end="...",flush=True)
	data = pd.read_csv(os.path.join("nsrdb",file),
		index_col=[0],
		usecols=outputs.keys(),
		parse_dates=[0],
		date_format="ISO8601")
	for source,output in outputs.items():
		if not output:
			continue
		if output not in results:
			results[output] = []
		try:
			results[output].append(pd.DataFrame({geocode:data[source].resample(resample).mean()}).round(precision).ffill().bfill())
		except Exception as err:
			print(source,"failed","-",err,end="!!!")
	print(f"{n+1} of {len(files)} done")

for output in outputs.values():
	if not output:
		continue
	results[output] = pd.concat(results[output],axis=1)
	results[output].index.name="timestamp"
	results[output].to_csv(os.path.join(output_folder,output+".csv"+(("."+compression) if compression else "")),index=True,header=True,compression=compression)
