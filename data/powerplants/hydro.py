import os, sys
sys.path.insert(0,"..")
import config
import pandas as pd

pd.options.display.max_columns = None
pd.options.display.width = None

INPUTS = {
    "GLOBALDATA" : "Global-Hydropower-Tracker-April-2024.xlsx",
}

OUTPUTS = {
    "PLANTDATA" : "hydro.csv",
}

STARTYEAR=2018
STOPYEAR=2022

data = pd.read_excel(INPUTS["GLOBALDATA"],sheet_name="Data")

mapper = {
    'Date Last Researched' : None, 
    'Country 1' : "country", 
    'Country 2' : None,
    'Project Name' : "name", 
    'Project Name (local lang/script)' : None, 
    'Other name(s)' : None, 
    'Capacity (MW)' : "capacity[MW]",
    'Binational' : None, 
    'Country 1 Capacity (MW)' : None, 
    'Country 2 Capacity (MW)' : None,
    'Turbines' : None, 
    'Status' : "status", 
    'Complex' : None, 
    'Start Year' : "start[y]", 
    'Retired Year' : "retirement[y]", 
    'Owner' : None,
    'Owner Name (local lang/script)' : None, 
    'Operator' : None,
    'Operator Name (local lang/script)' : None, 
    'Technology Type' : "type",
    'River / Watercourse' : None, 
    'Latitude' : "latitude", 
    'Longitude' : "longitude", 
    'Location Accuracy' : None,
    'City 1' : None, 
    'Local Area 1' : "county", 
    'Major Area 1' : None, 
    'State/Province 1' : "state",
    'Subregion 1' : None, 
    'Region 1' : None, 
    'City 2' : None, 
    'Local Area 2' : None, 
    'Major Area 2' : None,
    'State/Province 2' : None, 
    'Subregion 2' : None, 
    'Region 2' : None, 
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
data.drop(data[~data["state"].isin(config.state_list)].index,inplace=True)
data["unit"] = 1
data.to_csv(OUTPUTS["PLANTDATA"],header=True,index=False)