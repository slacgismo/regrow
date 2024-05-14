import os, sys
sys.path.insert(0,"..")
import config
import pandas as pd

pd.options.display.max_columns = None
pd.options.display.width = None

INPUTS = {
    "GLOBALDATA" : "Global-Coal-Plant-Tracker-January-2024.xlsx",
}

OUTPUTS = {
    "PLANTDATA" : "coal.csv",
}

STARTYEAR=2018
STOPYEAR=2022

data = pd.read_excel(INPUTS["GLOBALDATA"],sheet_name="Units")

mapper = {
    'GEM unit/phase ID' : None,
    'GEM location ID' : None,
    'Country' : "country",
    'Wiki URL' : None,
    'Plant name' : "name",
    'Unit name' : "unit",
    'Plant name (local)' : None, 
    'Plant name (other)' : None,
    'Owner' : None,
    'Parent' : None,
    'Capacity (MW)' : "capacity[MW]", 
    'Status' : "status",
    'Start year' : "start[y]",
    'Retired year' : "retirement[y]",
    'Planned retirement' : None,
    'Combustion technology' : None,
    'Coal type' : None,
    'Coal source' : None,
    'Alternate Fuel' : None,
    'Location' : None,
    'Local area (taluk, county)' : "county", 
    'Major area (prefecture, district)' : None,
    'Subnational unit (province, state)' : "state", 
    'Subregion' : None,
    'Region' : None,
    'Previous Region' : None,
    'Latitude' : "latitude",
    'Longitude' : "longitude",
    'Location accuracy' : None,
    'Permits (Detail)' : None, 
    'Permits (Main Date Only)' : None, 
    'Captive' : None,
    'Captive industry use' : None,
    'Captive residential use' : None,
    'Heat rate (Btu per kWh)' : None, 
    'Emission factor (kg of CO2 per TJ)' : None,
    'Capacity factor' : None,
    'Annual CO2 (million tonnes / annum)' : None,
    'Lifetime CO2' : None,
    'Remaining plant lifetime (years)' : None
    }

data.rename(dict([(x,y) for x,y in mapper.items() if not y is None]),inplace=True,axis=1)
data.drop([x for x,y in mapper.items() if y is None],axis=1,inplace=True)
data.drop(data[~data["status"].isin(["operating","retired"])].index,inplace=True)
data.drop(data[data["country"]!="United States"].index,inplace=True)
data.drop(data[data["retirement[y]"]<STARTYEAR].index,inplace=True)
data.drop(data[data["start[y]"].astype('int')>STOPYEAR].index,inplace=True)
data.drop(["country","status"],axis=1,inplace=True)
data.drop(data[~data["state"].isin(config.state_list)].index,inplace=True)
data["type"] = "ST"
data.to_csv(OUTPUTS["PLANTDATA"],header=True,index=False)
