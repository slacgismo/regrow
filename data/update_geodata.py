"""Update geodata files"""

import os, sys
import pandas as pd

outputs = {
	"Unnamed: 0" : None,
	"ghi" : "solar",
	"temp_air" : "temperature",
	}
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
			results[output].append(pd.DataFrame({geocode:data[source].resample("1h").mean()}).round(1).fillna(method='ffill'))
		except Exception as err:
			print(source,"failed","-",err,end="!!!")
	print(f"{n+1} of {len(files)} done")

for output in outputs.values():
	if not output:
		continue
	results[output] = pd.concat(results[output],axis=1)
	results[output].to_csv(os.path.join("geodata",output+".csv"),index=True,header=True)
