"""powerplants.py

Convert powerplants.csv data to GLM file to pypower model. Only operating
plants with capacity above 10 MW are included. Plants with "UNKNOWN" name
are modeled as anonymous (no name). Any column "NOT AVAILABLE" is omitted.
"""

minimum_capacity = 10.0 # minimum plant size to include
gridlabd_version = "-ge 4.3.7" # gridlabd version requirement

import sys, os
import pandas as pd
import datetime as dt
import re
import utils

if len(sys.argv) == 1:
    print("Syntax: python3 powerplants.py CSVFILE GLMFILE",file=sys.stderr)

csvname = sys.argv[1] if len(sys.argv) > 1 else "powerplants.csv.zip"
glmname = sys.argv[2] if len(sys.argv) > 2 else "powerplants.glm"

columns = dict([(y,x) for x,y in enumerate(pd.read_csv(csvname,nrows=2).columns)])

mapper = {
    "name" : "NAME",
    "latitude" : "LATITUDE",
    "longitude" : "LONGITUDE",
    "city" : "CITY",
    "state" : "STATE",
    "country" : "COUNTRY",
    "zipcode" : "ZIP",
    "naics_code" : "NAICS_CODE",
    "naics_description" : "NAICS_DESC",
    "plant_type" : "TYPE",
    "status" : "STATUS",
    "plant_code" : "PLANT_CODE",
    "operating_capacity" : "OPER_CAP",
    "summer_capacity" : "SUMMER_CAP",
    "winter_capacity" : "WINTER_CAP",
    "capacity_factor" : "CAP_FACTOR",
    "substation_1" : "SUB_1",
    "substation_2" : "SUB_2",
}

enumerated_types = {
    "plant_type" : [],
    "status" : [],
    "primary_fuel" : [],
    "secondary_fuel": [],
    }

set_types = ["plant_type"]

units = {
    "operating_capacity" : "MW",
    "summer_capacity" : "MW",
    "winter_capacity" : "MW",
    "capacity_factor" : "pu",
}

# construct args for full read of data
index_col = ["STATE"]
usecols = [columns[x] for x in index_col]
usecols.extend([columns[x] for x in mapper.values()])
data = pd.read_csv("powerplants.csv.zip",
                   index_col = index_col,
                   usecols = usecols,
                  ).sort_index()

if len(sys.argv) > 1 and sys.argv[1] == 'types': # generate list of plant types
    types = []
    for item in set(data["TYPE"]):
        for x in item.split("; "):
            if not x in types:
                types.append(x)
    print(types)
    quit()

plant_types = {
    # energy source and power generator
    'COAL INTEGRATED GASIFICATION COMBINED CYCLE' : ["COAL","CC"], 
    'NATURAL GAS FIRED COMBINED CYCLE' : ["NG", "CC"], 
    'NATURAL GAS FIRED COMBUSTION TURBINE' : ["NG","CT"], 
    'PETROLEUM COKE' : ["COKE","ST"], 
    'NATURAL GAS INTERNAL COMBUSTION ENGINE' : ["NG","IC"], 
    'NATURAL GAS STEAM TURBINE' : ["NG","ST"], 
    'PETROLEUM LIQUIDS' : ["OIL","ST"], 
    'OTHER GASES' : ["GAS","ST"], 
    'CONVENTIONAL STEAM COAL' : ["COAL","ST"], 
    'SOLAR PHOTOVOLTAIC' : ["SUN","PV"], 
    'BATTERIES' : ["ELEC","ES"], 
    'MUNICIPAL SOLID WASTE' : ["WASTE","CT"], 
    'NUCLEAR' : ["NUC","ST"], 
    'OTHER WASTE BIOMASS' : ["BIO","ST"], 
    'ONSHORE WIND TURBINE' : ["WIND","WT"], 
    'CONVENTIONAL HYDROELECTRIC' : ["WATER","HT"], 
    'WOOD/WOOD WASTE BIOMASS' : ["WOOD","ST"], 
    'FLYWHEELS' : ["ELEC","FW"], 
    'GEOTHERMAL' : ["GEO","ST"], 
    'OTHER NATURAL GAS' : ["NG","ST"], 
    'LANDFILL GAS' : ["WASTE","CT"], 
    'ALL OTHER' : ["OTHER","UNKNOWN"], 
    'NATURAL GAS WITH COMPRESSED AIR STORAGE' : ["ELEC","AT"], 
    'HYDROELECTRIC PUMPED STORAGE' : ["ELEC","HT"], 
    'SOLAR THERMAL WITHOUT ENERGY STORAGE' : ["SUN","ST"], 
    'SOLAR THERMAL WITH ENERGY STORAGE' : ["SUN","ES"], 
    'NOT AVAILABLE' : ["UNKNOWN","UNKNOWN"], 
    'OFFSHORE WIND TURBINE' : ["WIND","WT"],
}

if len(sys.argv) > 1 and sys.argv[1] == 'generator': # generate keywords for generator property
    print("""PT_set, "generator", get_generator_offset(), """)
    for value,keyword in enumerate(set([y for x,y in plant_types.values()])):
        print(f"""    PT_KEYWORD, "{keyword}", (set)0x{2**value:08x}, """)
    quit()

if len(sys.argv) > 1 and sys.argv[1] == 'fuel': # generate keywords for fuel property
    print("""PT_set, "fuel", get_fuel_offset(), """)
    for value,keyword in enumerate(set([x for x,y in plant_types.values()])):
        print(f"""    PT_KEYWORD, "{keyword}", (set)0x{2**value:08x}, """)
    quit()

zipranges = {
    # USA
    "AZ" : ["85001","86556"],
    "CA" : ["90001","96162"],
    "CO" : ["80001","81658"],
    "ID" : ["83201","83876"],
    "MT" : ["59001","59937"],
    "NM" : ["87001","88441"],
    "OR" : ["97001","97920"],
    "UT" : ["84001","84784"],
    "WA" : ["98001","99403"],
    "WY" : ["82001","83128"],
    # Canada
    "AB" : [], # Alberta
    "BC" : [], # British Columbia
    # Mexico
    "BA" : [], # Baja California
    "SO" : [], # Sonora
}

gencosts = pd.read_csv("../data/gencosts.csv",index_col=['powerplant'])
nodesgis = pd.read_csv("wecc240_gis.csv")
nodesgis["geocode"] = [utils.geohash(float(x),float(y)) for x,y in nodesgis[["Lat","Long"]].values]
nodesgis = nodesgis.set_index(["geocode","Base kV"]).sort_index().reset_index().set_index("geocode")
nodesgis = nodesgis[~nodesgis.index.duplicated(keep='first')]

with open(glmname,'w') as glm:
    print(f"// generated from {csvname} at {dt.datetime.now()}",file=glm)
    if gridlabd_version:
        print("#version",gridlabd_version,file=glm)
    print("module pypower;",file=glm)
    count = 0
    names = []
    for _state,_ziprange in zipranges.items():
        if _ziprange:
            for _n,_plant in data.loc[_state].reset_index().sort_values('OPER_CAP',ascending=False).iterrows():
                genname = None
                if _plant['OPER_CAP'] < minimum_capacity:
                    # print(f"WARNING [powerplant.py]: powerplant '{_plant['NAME']}' is less than minimum_capacity of {minimum_capacity} MW",file=sys.stderr)
                    break # plants are ordered by size in each state
                if _plant['STATUS'] != "OP":
                    if _plant['STATUS'] == "NOT AVAILABLE":
                        print(f"WARNING [powerplant.py]: powerplant '{_plant['NAME']}' status is unknown and assumed OFFLINE",file=sys.stderr)
                    else:
                        continue
                if len(_ziprange) > 1 and _plant['ZIP'] != "NOT AVAILABLE" and ( _plant['ZIP'] < _ziprange[0] or _plant['ZIP'] > _ziprange[1] ):
                    print(f"WARNING [powerplant.py]: powerplant '{_plant['NAME']}' zipcode '{_plant['ZIP']}' is not valid for state '{_plant['STATE']}'",file=sys.stderr)

                print("object powerplant {",file=glm)
                fuel = []
                generator = []
                location = {}
                for name,column in mapper.items():
                    value = _plant[column]
                    if column == "NAME":
                        if value == "UNKNOWN":
                            continue;
                        if value in names:
                            try:
                                value = value[:-1] + str(int(value[-1])+1)
                            except:
                                value += " 2"
                        names.append(value)
                        value = value.replace(" ","_").replace("(","").replace(")","").replace(",","").replace("/","_").replace("&","").replace("__","_").replace(".","").replace("'","").replace("#","")
                        if '0' <= value[0] <= '9':
                            value = "_" + value
                        if len(value) > 63:
                            print(f"WARNING [powerplant.py]: plant name '{value}' is too long",file=sys.stderr)
                    if value != "NOT AVAILABLE" or name in ['status']:
                        if name == "plant_type":
                            for item in value.split("; "):
                                fuel.append(plant_types[item][0])
                                generator.append(plant_types[item][1])
                            print(f"""    generator {"|".join(set(generator))};""",file=glm)
                            print(f"""    fuel {"|".join(set(fuel))};""",file=glm)
                        elif name == "status":
                            print(f"""    status "{"ONLINE" if _plant['STATUS'] == "OP" else "OFFLINE"}";""",file=glm)
                        elif name in units:
                            print(f"""    {name} {value} {units[name]};""",file=glm)
                        elif name in ["latitude","longitude"]:
                            location[name] = value
                            print(f"""    {name} {value:.4f};""",file=glm)
                        elif name in ["naics_description","city","substation_1","substation_2"]:
                            print(f"""    {name} "{value.title()}";""",file=glm)    
                        else:
                            print(f"""    {name} "{value}";""",file=glm)
                            if name == "name":
                                genname = value
                        
                if "latitude" in location and "longitude" in location:
                    node = utils.nearest(utils.geohash(float(location["longitude"]),float(location["latitude"])),nodesgis.index)
                    print(f"""    parent "wecc240_psse_G_{nodesgis.loc[node]['Bus  Number']}_0";""",file=glm)
                else:
                    print(f"WARNING [powerplants.py]: powerplant {genname} is missing location data")

                if not genname in gencosts.index:
                    gencosts = pd.concat([gencosts,pd.DataFrame(
                                            {
                                                "powerplant" : [genname],
                                                "generator" : ["|".join(generator)],
                                                "fuel": ["|".join(fuel)],
                                                "startup" : [0.0],
                                                "shutdown" : [0.0],
                                                "model" : [1],
                                                "costs" : ["0,0"],
                                            },
                                            ).set_index("powerplant")]).reset_index().set_index(["powerplant"]).sort_index()
                    print(f"WARNING [powerplants.py]: added {genname} to gencosts.csv")

                print(f"""    object gencost 
    {{
        name "PC_{genname}";
        startup 0;
        shutdown 0;
        model 2;
        costs "0.01,40,0";
    }};""",file=glm)
                print("}", file=glm)
                count += 1

    print(f"{count} generators found")

gencosts.to_csv("gencosts.csv",index=True,header=True)
