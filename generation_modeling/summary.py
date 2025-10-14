"""Import HFLD powerplant data"""

import pandas as pd

FILE = "powerplants.csv.zip"
COLUMNS = [
    "NAME","PRIM_FUEL",
    "OPER_CAP","SUMMER_CAP","WINTER_CAP",
    "COAL_USED","NGAS_USED",
    "LATITUDE","LONGITUDE",
    "COUNTYFIPS","STATUS",
    "CITY","STATE","ZIP",
    ]

WECC = ["CA","WA","OR","ID","UT","NV","AZ","NM","WY","CO","MT"]

plant_fuel = {
    "AB": "Agricultural byproduct",
    "BFG": "Blast furnace gas",
    "BIT": "Bituminous coal",
    "BLQ": "Black liquor",
    "COG": "Coke oven gas",
    "DFO": "Distillate fuel oil, light fuel oil, diesel oil",
    "GEO": "Geothermal",
    "H2": "Hydrogen",
    "JF": "Jet fuel",
    "KER": "Kerosene",
    "LFG": "Landfill gas",
    "LIG": "Lignite coal",
    "MSW": "Municipal solid waste",
    "MWH": "Electricity used for energy storage (megawatt hour)",
    "NG": "Natural gas",
    "NUC": "Nuclear",
    "OBG": "Other biomass gas (digester gas, methane, and other biomass gases)",
    "OBL": "Other biomass liquids",
    "OBS": "Other biomass solid",
    "OG": "Other gas",
    "OTH": "Other",
    "PC": "Petroleum coke",
    "PG": "Gaseous propane",
    "PRG": "Process gas",
    "PUR": "Purchased steam",
    "RC": "Refined coal",
    "RFO": "Residual fuel oil, heavy fuel oil, petroleum",
    "RG": "Refinery gas",
    "SGC": "Coal-derived synthetic gas",
    "SLW": "Sludge waste",
    "SUB": "Subbituminous coal",
    "SUN": "Solar",
    "TDF": "Tire-derived fuel",
    "WAT": "Water",
    "WC": "Waste coal",
    "WDL": "Wood, wood waste liquid",
    "WDS": "Wood, wood waste solid",
    "WH": "Waste heat",
    "WND": "Wind",
    "WO": "Waste oil",
}

fuel_types = {
    "gas" : ["NG","OG","PG","RG","SGC",],
    "geo": ["GEO",],
    "liquid": ["BLQ","DFO","JF","KER","RFO",],
    "nuclear": ["NUC",],
    "solar": ["SUN",],
    "solid" : ["BIT","LIG","RC","SUB"],
    "steam": ["PUR",],
    "storage": ["MWH"],
    "waste": ["AB","BFG","COG","LFG","MSW","OBG","OBL","OBS","PC","SLW","TDF","WC","WDL","WDS","WH","WO"],
    "water": ["WAT"],
    "wind": ["WIND",],
}

type_fuels = {}
for x,y in fuel_types.items():
    for z in y:
        type_fuels[z] = x

def powerplants(
        file:str=FILE,
        columns:list[str]=COLUMNS,
        states:list[str]=WECC,
        mincap:float=0,
        status:list[str]=["OP"]
        ) -> pd.DataFrame:
    """Read powerplant data"""
    data = pd.read_csv(file,
        usecols=COLUMNS,
        )
    data.drop(data[(~data.STATE.isin(WECC))|(data.OPER_CAP<=mincap)|(data.STATUS.isin(status))].index,inplace=True)
    return data

def summary(*args,**kwargs):
    """Summary powerplant data"""
    data = powerplants(*args,**kwargs)

    data["FUEL_TYPE"] = [type_fuels[x.PRIM_FUEL] if x.PRIM_FUEL in type_fuels else None for n,x in data.iterrows()]

    data["WINTER_CAP"] = [x.WINTER_CAP if x.WINTER_CAP > 0 else x.OPER_CAP for n,x in data.iterrows()]
    data["SUMMER_CAP"] = [x.SUMMER_CAP if x.SUMMER_CAP > 0 else x.OPER_CAP for n,x in data.iterrows()]

    plants = data.groupby("FUEL_TYPE")
    result = pd.concat([
        pd.DataFrame(plants["OPER_CAP"].count()),
        pd.DataFrame(round(plants["OPER_CAP"].sum()/1000,3)),
        pd.DataFrame(round(plants["WINTER_CAP"].sum()/1000,3)),
        pd.DataFrame(round(plants["SUMMER_CAP"].sum()/1000,3)),
        ],axis=1)
    result.columns = ["count","operating_capacity[GW]","winter_capacity[GW]","summer_capacity[GW]"]

    return result

if __name__ == '__main__':

    pd.options.display.width=None
    pd.options.display.max_columns=None

    print(summary().sort_values(["operating_capacity[GW]"],ascending=False))

