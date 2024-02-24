"""powerplants.py

Convert powerplants.csv data to GLM file to pypower model. Only operating
plants with capacity above 10 MW are included. Plants with "UNKNOWN" name
are model as anonymous (no name). Any column "NOT AVAILABLE" is omitted.
"""
import sys, os
import pandas as pd
import datetime as dt

if len(sys.argv) < 3:
    print("Syntax: python3 powerplants.py CSVFILE GLMFILE")

csvname = sys.argv[1]
glmname = sys.argv[2]

minimum_capacity = 10.0 # minimum plant size to include

columns = dict([(y,x) for x,y in enumerate(pd.read_csv(csvname).columns)])
# for name,column in columns.items():
#     print(f"{column:2d} {name:20s}")
mapper = {
    "name" : "NAME",
    "latitude" : "LATITUDE",
    "longitude" : "LONGITUDE",
    "city" : "CITY",
    "state" : "STATE",
    "zipcode" : "ZIP",
    "country" : "COUNTRY",
    "naics_code" : "NAICS_CODE",
    "naics_description" : "NAICS_DESC",
    "plant_type" : "TYPE",
    "status" : "STATUS",
    "plant_code" : "PLANT_CODE",
    "operating_capacity" : "OPER_CAP",
    "summer_capacity" : "SUMMER_CAP",
    "winter_capacity" : "WINTER_CAP",
    "capacity_factor" : "CAP_FACTOR",
    "primary_fuel" : "PRIM_FUEL",
    "secondary_fuel" : "SEC_FUEL",
    "substation_1" : "SUB_1",
    "substation_2" : "SUB_2",
}
# print([columns[x] for x in mapper.values()])

index_col = ["STATE"]
usecols = [columns[x] for x in index_col]
usecols.extend([columns[x] for x in mapper.values()])
data = pd.read_csv("powerplants.csv.zip",
                   index_col = index_col,
                   usecols = usecols,
                  ).sort_index()

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

with open(glmname,'w') as glm:
    print(f"// generated from {csvname} at {dt.datetime.now()}",file=glm)
    print("module pypower;",file=glm)
    count = 0
    for _state,_ziprange in zipranges.items():
        if _ziprange:
            for _n,_plant in data.loc[_state].reset_index().sort_values('OPER_CAP',ascending=False).iterrows():
                if _plant['OPER_CAP'] < minimum_capacity or _plant['STATUS'] != "OP":
                    break
                print("object generator {",file=glm)
                for name,column in mapper.items():
                    if column == "NAME" and _plant[column] == "UNKNOWN":
                        continue;
                    if _plant[column] != "NOT AVAILABLE":
                        print(f"""    {name} "{_plant[column]}";""",file=glm)
                print("}", file=glm)
                count += 1

    print(f"{count} generators found")
