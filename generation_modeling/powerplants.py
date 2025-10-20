"""WECC powerplant GIS model"""

import os
import datetime as dt
import pandas as pd
import utils

pd.options.display.width = None
pd.options.display.max_columns = None

WECC = ["CA","WA","OR","ID","UT","NV","AZ","NM","WY","CO","MT"]

FUEL = {
    'AB': "BIO",
    'BIT': "COAL",
    'BLQ': "COKE",
    'DFO': "OIL",
    'GEO': "GEO",
    'JF': "OIL",
    'LFG': "WASTE",
    'MSW': "WASTE",
    'MWH': "ELEC",
    'NG': "GAS",
    'NUC': "NUC",
    'OBG': "BIO",
    'OG': "GAS",
    'OTH': "OTHER",
    'PC': "COKE",
    'PUR': "OTHER",
    'RC': "GAS",
    'SUB': "COAL",
    'SUN': "SUN",
    'WAT': "WATER",
    'WC': "WASTE",
    'WDS': "WASTE",
    'WH': "WASTE",
    'WND': "WIND",
}

GENERATOR = {
    'SOLAR': "PV",
    'OTHF': "UNKNOWN",
    'GAS': "CC",
    'COAL': "ST",
    'HYDRO': "HT",
    'OIL': "CT",
    'WIND': "WT",
    'BIOMASS': "CT",
    'NUCLEAR': "ST",
    'GEOTHERMAL': "ST",
    'OFSL': "IC",
}

COST = {
    "*": { # default costs
        'SOLAR': [0,0,0],
        'OTHF': [0,0,0],
        'GAS': [0,0,0],
        'COAL': [0,0,0],
        'HYDRO': [0,0,0],
        'OIL': [0,0,0],
        'WIND': [0,0,0],
        'BIOMASS': [0,0,0],
        'NUCLEAR': [0,0,0],
        'GEOTHERMAL': [0,0,0],
        'OFSL': [0,0,0],
    },
}

GENTYPE = {
    'SOLAR': "PV-2",
    'OTHF': "Steam",
    'GAS': "Gas",
    'COAL': "Steam",
    'HYDRO': "Hydro-3",
    'OIL': "Steam",
    'WIND': "Wind-2",
    'BIOMASS': "Biomass",
    'NUCLEAR': "Nuclear",
    'GEOTHERMAL': "Geothermal",
    'OFSL': "Steam",
    }

def get_costs(refresh=True):
    """Load new powerplant construction/production costs

    Arguments:

    - refresh: force download from google sheets
    """
    docid,sheet = "1dLvUglBP2ojGRXTCQGKJ4BlkiJQfcrGxZNu9kLjZpKI","Sheet1"
    options = dict()
    if os.path.exists("generation_cost.csv") and not refresh:
        return pd.read_csv("generation_cost.csv",index_col=[0,1],header=0)
    else:
        data = pd.read_csv(f"https://docs.google.com/spreadsheets/d/{docid}/gviz/tq?tqx=out:csv&sheet={sheet}",
            index_col=[0,1],header=0)
        data.to_csv("generation_cost.csv",index=True,header=True)
        return data

def get_powerplants(source="EIA",states=WECC):
    """Load powerplant database"""
    COLUMNS = {
        "EIA":{
            "SEQPLT22": "id",
            "PSTATABB": "state",
            "PNAME": "name",
            "ORISPL": "code",
            "FIPSST": "state_fips",
            "FIPSCNTY": "county_fips",
            "CNTYNAME": "county",
            "LAT": "latitude",
            "LON": "longitude",
            "PLPRMFL": "fuel",
            "PLFUELCT": "generator",
            "NAMEPCAP": "capacity",
        },
    }
    assert source in COLUMNS, f"source {source} is not supported"

    egrid = pd.read_excel("egrid2022_data.xlsx",sheet_name="PLNT22",
        skiprows=1,
        usecols=COLUMNS[source].keys(),
        dtype={"FIPSST":str,"FIPSCNTY":str,"PLPRMFL":str,"PLFUELCT":str}
        )
    egrid.drop(egrid[~egrid["PSTATABB"].isin(states)].index,inplace=True)
    egrid.rename(COLUMNS[source],axis=1,inplace=True)
    egrid["fips"] = [f"{x.state_fips}{x.county_fips}" for _,x in egrid.iterrows()]
    egrid.drop(["state_fips","county_fips"],axis=1,inplace=True)

    return egrid

# read cost data
costs = get_costs()

# read GIS data from NREL
weccgis = pd.read_csv("wecc240_gis.csv")
weccgis["geocode"] = [utils.geohash(x["Lat"],x["Long"]) for _,x in weccgis.iterrows()]
weccgis.set_index("geocode",inplace=True)

def get_nearest(latitude,longitude,gentype):
    """Find the nearest bus to a lat/lon"""
    geohash = utils.geohash(latitude,longitude)
    closest = utils.nearest(geohash,gendata.loc[gentype].index)
    assert closest in weccgis.index, f"{closest=} not found in weccgis.index"

    return closest

egrid = get_powerplants("EIA")

gencost = pd.concat([pd.read_csv("gen.csv"),pd.read_csv("gencost.csv")],axis=1).set_index("GEN_BUS").join(weccgis.reset_index().set_index("Bus  Number")).reset_index()
gendata = pd.read_csv("generation_data.csv",header=0,usecols=["genname","Gen_Type"]).join(gencost).set_index(["Gen_Type","geocode"]).sort_index()

counties = pd.read_csv("counties.csv")
gentypes = {x:y["type"].upper() for x,y in pd.read_csv("generation_types.csv",index_col=["id"]).to_dict("index").items()}

def get_glm(id,state,name,code,fips,county,latitude,longitude,fuel,generator,capacity):

    gentype = GENTYPE[generator if isinstance(generator,str) else "OTHF"]
    geohash = get_nearest(latitude,longitude,gentype)
    location = weccgis.loc[geohash]
    if "Bus  Number" in location.index:
        bus_number = location['Bus  Number']
        bus_name = location['Bus  Name']
    else: # multiple busses found
        bus_number = location.iloc[0]['Bus  Number']
        bus_name = location.iloc[0]['Bus  Name']

    data = gendata.loc[gentype,geohash].iloc[0]
    match data.NCOST:
        case 1:
            fixed_cost = data.COST0
            variable_cost = data.COST1
            scarcity_cost = data.COST0
        case 2:
            fixed_cost = data.COST1
            variable_cost = data.COST0
            scarcity_cost = data.COST0
        case 3:
            fixed_cost = data.COST2
            variable_cost = data.COST1
            scarcity_cost = data.COST0
        case _:
            fixed_cost = variable_cost = scarcity_cost = 0.0

    return f"""object powerplant
{{
    name "{name}";
    parent "{f"wecc240_psse_G_{bus_number}"}"; // {bus_name} ({geohash})
    latitude "{latitude}";
    longitude "{longitude}";
    county "{county} County";
    state "{state}";
    country "USA";
    plant_code "{code}";
    generator {GENERATOR[generator] if generator in GENERATOR else "UNKNOWN"};
    fuel {FUEL[fuel] if fuel in FUEL else "UNKNOWN"};
    status ONLINE;
    operating_capacity {capacity} MW;
    fixed_cost {fixed_cost} $/h;
    variable_cost {variable_cost} $/MWh;
}}
"""

if __name__ == '__main__':

    pd.options.display.width=None
    pd.options.display.max_columns=None

    with open("powerplants.glm","w") as fh:
        print(f"// generated by {__file__} at {dt.datetime.now()}",file=fh)
        print("module pypower;",file=fh)
        for plant_id,plant_data in egrid.iterrows():
            print(get_glm(**plant_data.to_dict()),file=fh)

    # quit()
    # gisdata = gendata.set_index("busname").join(weccgis.set_index("Bus  Number")).reset_index()
    # gisdata["gentype"] = [gentypes[x["genname"].replace(str(x["busname"]),"")] for n,x in gisdata.iterrows()]
    # print(sorted(gisdata.gentype.unique()))