"""Check geodata"""

import sys
import os
import json
sys.path.append("../data")
import utils

geodata = "../data/geodata"
model = json.load(open("wecc240.json","r"))

def find(property:str,value:str) -> dict:
    """Find objects having a property value"""
    return {x:y for x,y in model["objects"].items() if property in y and y[property] == value}

def columns(file:str) -> list:
    """Return columns in csv file"""
    return [x.strip() for x in open(file,"r").readlines()[0].split(",")]

def getbus(obj:str) -> str:
    data = model["objects"][obj]
    if data["class"] == "bus":
        return data
    elif "parent" in data:
        return getbus(data["parent"])
    raise ValueError(f"object '{obj}' is not associated with a bus")

geofile = {}
for file in os.listdir(geodata):
    if file.endswith(".csv"):
        geofile[file] = columns(os.path.join(geodata,file))[1:]
print(len(geofile),"geodata files found")

needed = {
    "load":["heating.csv","cooling.csv","total.csv"],
    "gen":["pv.csv","wt.csv"],
    "powerplant":["wind.csv","solar.csv"],
}

checks = 0
errors = 0
missing = {}

for x,y in needed.items():
    for z in y:
        if not z in geofile:
            missing[z] = ["no geodata found"]
            checks += 1
            errors += 1

for oclass in ["load","gen","powerplant"]:
    objects = find("class",oclass)
    for obj,data in objects.items():
        if "parent" in data:
            data = getbus(data["parent"])
        elif "bus" in data:
            data = find("bus_i",data["bus"])
        else:
            print("ERROR [check_geodata.py]:",obj,"has no bus associated with it",file=sys.stderr)
        if "latitude" in data and "longitude" in data:
            geohash = utils.geohash(float(data["latitude"]),float(data["longitude"]))
            for file,columns in [(x,y) for x,y in geofile.items() if x in needed[oclass]]:
                if geohash not in columns:
                    print("ERROR [check_geodata.py]:",oclass,obj,geohash,"not found in",file,file=sys.stderr)
                    errors += 1
                    if file not in missing:
                        missing[file] = []
                    if geohash not in missing[file]:
                        missing[file].append(geohash)
                checks +=1

print(checks,"checks completed")
print(errors,"errors found")
if errors > 0:
    print("Summary of missing geohashes")
    for file,data in missing.items():
        print(file," ".join(data),sep=": ")
