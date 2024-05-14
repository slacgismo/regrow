import os, sys
sys.path.insert(0,"..")
import config
import pandas as pd

pd.options.display.max_columns = None
pd.options.display.width = None

INPUTS = {
    "GLOBALDATA" : "Global-Oil-and-Gas-Plant-Tracker-GOGPT-February-2024-v4.xlsx",
}

OUTPUTS = {
    "GASDATA" : "gas.csv",
    "OILDATA" : "oil.csv",
}

STARTYEAR=2018
STOPYEAR=2022

data = pd.read_excel(INPUTS["GLOBALDATA"],sheet_name="Gas & Oil Units")

mapper = {
    'Wiki URL' : None,
    'Country' : "country",
    'Plant name' : "name",
    'Plant name (local script)' : None,
    'Unit name' : "unit",
    'Fuel' : "fuel",
    'Capacity (MW)' : "capacity[MW]", 
    'Status' : "status",
    'Technology' : "type",
    'CHP' : None,
    'Hydrogen capable?' : None, 
    'CCS attachment?' : None,
    'Coal-to-gas conversion/replacement?' : None, 
    'Start year' : "start[y]",
    'Retired year' : "retirement[y]",
    'Planned retire' : None,
    'Operator' : None,
    'Owner' : None,
    'Parent' : None,
    'Latitude' : "latitude",
    'Longitude' : "longitude",
    'Location accuracy' : None,
    'City' : None,
    'Local area (taluk/county)' : "county",
    'Major area (prefecture/district)' : None, 
    'Subnational unit (province/state)' : "state",
    'Region' : None,
    'Sub-region' : None,
    'Other IDs (location)' : None, 
    'Other IDs (unit)' : None,
    'Other plant names' : None,
    'Captive [heat/power/both]' : None,
    'Captive industry type' : None,
    'Captive non-industry use [heat/power/both/none]' : None, 
    'GEM location ID' : None,
    'GEM unit ID' : None,
    }

gasfuels = ["NG","LNG","WSTH-NG","LFG","BG","BFG","COG","LPG","OG"]

def isgas(fuels):
    return len([x for x in fuels.split("|") if x in gasfuels])>0

data.rename(dict([(x,y) for x,y in mapper.items() if not y is None]),inplace=True,axis=1)
data.drop([x for x,y in mapper.items() if y is None],axis=1,inplace=True)
data.drop(data[~data["status"].isin(["operating","retired"])].index,inplace=True)
data.drop(data[data["country"]!="United States"].index,inplace=True)
data.drop(data[data["retirement[y]"]<STARTYEAR].index,inplace=True)
data.drop(data[data["start[y]"].astype('int')>STOPYEAR].index,inplace=True)
data.drop(["country","status"],axis=1,inplace=True)
data.drop(data[~data["state"].isin(config.state_list)].index,inplace=True)
data["type"][data["type"]=="GT"] = "CT"

oilfuel = [not isgas(x) for x in data["fuel"]]
oil = data.loc[oilfuel]
oil.to_csv(OUTPUTS["OILDATA"],header=True,index=False)

gasfuel = [isgas(x) for x in data["fuel"]]
gas = data.loc[gasfuel]
gas.to_csv(OUTPUTS["GASDATA"],header=True,index=False)
