import os, sys
sys.path.insert(0,"..")
import config
import pandas as pd

pd.options.display.max_columns = None
pd.options.display.width = None

INPUTS = {
    "GLOBALDATA" : "Global-Solar-Power-Tracker-December-2023.xlsx",
}

OUTPUTS = {
    "PLANTDATA" : "solar.csv",
}

STARTYEAR=2018
STOPYEAR=2022

data = pd.read_excel(INPUTS["GLOBALDATA"],sheet_name="Large Utility-Scale")

mapper = {
    'Date Last Researched' : None,
    'Country' : "country",
    'Project Name' : "name",
    'Phase Name': "unit",
    'Project Name in Local Language / Script' : None,
    'Other Name(s)': None,
    'Capacity (MW)': "capacity[MW]", 
    'Technology Type' : None,
    'Status' : "status",
    'Start year': "start[y]",
    'Retired year' : "retirement[y]",
    'Operator' : None,
    'Operator Name in Local Language / Script': None,
    'Owner' : None,
    'Owner Name in Local Language / Script' : None,
    'Latitude': "latitude",
    'Longitude' : "longitude",
    'Location accuracy' : None,
    'City' : None,
    'Local area (taluk, county)': "county",
    'Major area (prefecture, district)': None, 
    'State/Province' : "state",
    'Subregion': None,
    'Region' : None,
    'GEM location ID' : None,
    'GEM phase ID' : None,
    'Other IDs (location)': None,
    'Other IDs (unit/phase)': None, 
    'Wiki URL': None
    }

data.rename(dict([(x,y) for x,y in mapper.items() if not y is None]),inplace=True,axis=1)
data.drop([x for x,y in mapper.items() if y is None],axis=1,inplace=True)
data.drop(data[~data["status"].isin(["operating","retired"])].index,inplace=True)
data.drop(data[data["country"]!="United States"].index,inplace=True)
data.drop(data[data["retirement[y]"]<STARTYEAR].index,inplace=True)
data.drop(data[data["start[y]"]>STOPYEAR].index,inplace=True)
data.drop(["country","status"],axis=1,inplace=True)
data.drop(data[~data["state"].isin(config.state_list)].index,inplace=True)
data["type"] = "PV"
data.to_csv(OUTPUTS["PLANTDATA"],header=True,index=False)