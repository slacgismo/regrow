import pandas as pd
import datetime as dt

INPUTS={
	"CA-NW" : "https://raw.githubusercontent.com/enliten/ENLITEN-Grid-Econ-Data/main/Source%20Data/Power%20Flow%20Interchange/Source_Interchange_CA_NW.csv",
	"CA-SW" : "https://raw.githubusercontent.com/enliten/ENLITEN-Grid-Econ-Data/main/Source%20Data/Power%20Flow%20Interchange/Source_Interchange_CA_SW.csv",
}
OUTPUTS={
	"CA-NW" : "CA-NW.csv",
	"CA-SW" : "CA-SW.csv",
}

for item in INPUTS.keys():
	data = pd.read_csv(INPUTS[item],skiprows=4,header=0,index_col=[0])
	data.sort_index(inplace=True)
	data.index = pd.to_datetime([x.replace("H",":00:00+00:00") for x in data.index],format="%m/%d/%Y %H:%M:%S%z")
	data.index = pd.to_datetime(data.index,utc=True)
	data.index.name = "datetime"
	data.columns = ["power[MW]"]
	data.to_csv(OUTPUTS[item],header=True,index=True)
