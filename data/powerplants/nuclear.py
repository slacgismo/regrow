import pandas as pd

pd.options.display.max_columns = None
pd.options.display.width = None

INPUTS = {
	"GLOBALDATA" : "Global-Nuclear-Power-Tracker-October-2023.xlsx",
}

OUTPUTS = {
	"PLANTDATA" : "nuclear.csv",
}

STARTYEAR=2018
STOPYEAR=2022

data = pd.read_excel(INPUTS["GLOBALDATA"],sheet_name="Data")

mapper = {
	'Date Last Researched' : None, 
	'Country' : "country", 
	'Project Name' : "name", 
	'Unit Name' : "unit",
	'Project Name in Local Language / Script' : None, 
	'Other Name(s)' : None,
	'Capacity (MW)' : "capacity[MW]", 
	'Status' : "status", 
	'Reactor Type' : "type", 
	'Model' : None, 
	'Start Year' : "start[y]",
	'Retirement Year' : "retirement[y]", 
	'Planned Retirement' : None, 
	'Cancellation Year' : None,
	'Construction Start Date' : None, 
	'First Criticality Date' : None,
	'First Grid Connection' : None, 
	'Commercial Operation Date' : None, 
	'Retirement Date' : None,
	'Owner' : None, 
	'Owner Name in Local Language / Script' : None, 
	'Operator' : None,
	'Operator Name in Local Language / Script' : None,
	'Reference Net Capacity (MW)' : None, 
	'Design Net Capacity (MW)' : None,
	'Thermal Capacity (MWt)' : None, 
	'Latitude' : "latitude", 
	'Longitude' : "longitude", 
	'Location Accuracy' : None,
	'City' : None, 
	'Local Area (taluk, county)' : "county",
	'Major Area (prefecture, district)' : None, 
	'State/Province' : "state", 
	'Subregion' : None,
	'Region' : None, 
	'GEM location ID' : None, 
	'GEM unit ID' : None, 
	'Wiki URL' : None,
	}

data.rename(dict([(x,y) for x,y in mapper.items() if not y is None]),inplace=True,axis=1)
data.drop([x for x,y in mapper.items() if y is None],axis=1,inplace=True)
data.drop(data[~data["status"].isin(["operating","retired"])].index,inplace=True)
data.drop(data[data["country"]!="United States"].index,inplace=True)
data.drop(data[data["retirement[y]"]<STARTYEAR].index,inplace=True)
data.drop(data[data["start[y]"]>STOPYEAR].index,inplace=True)
data.drop(["country","status"],axis=1,inplace=True)
data.to_csv(OUTPUTS["PLANTDATA"],header=True,index=False)
