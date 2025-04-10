"""Check geodata"""

import sys
import os
import json
import math
import numpy as np
sys.path.append("../data")
import utils
import pandas as pd

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
            for file,hashes in [(x,y) for x,y in geofile.items() if x in needed[oclass]]:
                if geohash not in hashes:
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
    print("Potentially missing geohashes")
    for file,data in missing.items():
        print(file," ".join(data),sep=": ")

geolist = []
for x in geofile.values():
    geolist += x
geolist = list(set(geolist))
buslist = []
baseMVA = float(model["globals"]["pypower::baseMVA"]["value"].split()[0])
for bus,data in find("class","bus").items():
    baseKV = float(data["baseKV"].split()[0])*1000
    baseZ = baseKV**2/baseMVA
    if "latitude" in data and "longitude" in data:
        lat = float(data["latitude"])
        lon = float(data["longitude"])
        geohash = utils.geohash(lat,lon)
        nearest = utils.nearest(geohash,geolist)
        distance = utils.distance(geohash,nearest)*40000*math.cos(lat*math.pi/180)/360 # very rough conversion to km
        
        load = np.array([abs(complex(y["S"].split()[0])) for x,y in find("class","load").items() if y["parent"] == bus]).sum()
        if load == 0:
            P = np.array([abs(complex(y["P"].split()[0])) for x,y in find("class","load").items() if y["parent"] == bus])
            I = np.array([abs(complex(y["I"].split()[0])) for x,y in find("class","load").items() if y["parent"] == bus])
            Z = np.array([abs(complex(y["Z"].split()[0])) for x,y in find("class","load").items() if y["parent"] == bus])
            load = (Z + I + P).sum()
    
        gen = [x for x,y in find("class","gen").items() if y["bus"] == data["bus_i"]]
        wind = solar = generation = storage = float('nan')
        if len(gen) > 0:
            wind = solar = generation = storage = 0
            for plant,data in [(x,y) for x,y in find("class","powerplant").items() if y["parent"] in gen]:
                capacity = float(data["operating_capacity"].split()[0])
                if data["generator"] == "PV":
                    solar += capacity
                elif data["generator"] == "WT":
                    wind += capacity
                elif data["generator"] == "ES":
                    storage += capacity
                else:
                    generation += capacity

        buslist.append(pd.DataFrame(data={
            "bus": [bus.split("_")[-1]],
            "geocode": [geohash],
            "wind[MVA]": [wind],
            "solar[MVA]": [solar],
            "generation[MVA]":[generation],
            "load[MVA]":[load],
            "max_import[%]":[load-generation]/load*100 if load>0 else float('nan'),
            "min_import[%]":[load-generation-wind-solar]/load*100 if load>0 else float('nan'),
            "storage[MVA]":[storage],
            "nearest":[nearest if nearest != geohash else ""],
            "distance[km]":[round(distance,1) if nearest != geohash else ""],
            "total.csv": ["MISSING" if not nearest in geofile["total.csv"] and load>0 else "OK"],
            "pv.csv": ["MISSING" if not nearest in geofile["pv.csv"] and solar>0 else "OK"],
            "wt.csv": ["MISSING" if not nearest in geofile["wt.csv"] and wind>0 else "OK"],
            }))
pd.concat(buslist).round(1).to_csv("check_geodata.csv",index=False,header=True)
