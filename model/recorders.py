"""WECC 240 model recorders"""

import os
import sys
import datetime as dt
import math

import geotape

loads_recorder = None
plant_recorder = None

def voltage_pu(data):
    return f"{abs(data['V']) / data['Vn']:.3f}"

def voltage_deg(data):
    V = data['V']
    return f"{math.atan2(V.imag,V.real):.3f}"

def status(x):
    return ["OFFLINE","ONLINE","CURTAILED"][x]

def generator(x):
    keywords = dict(CC=1024, PV=512, CT=256, ES=128, WT=64, FW=32, IC=16, AT=8, ST=4, HT=2, UNKNOWN=1)
    return "|".join([y for y,z in keywords.items() if x&z])

def fuel(x):
    keywords = dict(NG=32768, COAL=16384, WATER=8192, NUC=4096, GAS=2048, OTHER=1024, WOOD=512, UNKNOWN=256, OIL=128, BIO=64, WASTE=32, COKE=16, GEO=8, SUN=4, WIND=2, ELEC=1)
    return "|".join([y for y,z in keywords.items() if x&z])

def on_init(t0):

    global loads_recorder
    loads_recorder = geotape.GeoRecorder(gldcore,
        csvname="loads.csv",
        objects=r"^wecc240_psse_L_[0-9]+",
        properties={
            "status" : status,
            "S": lambda x: f"{abs(x):.1f}",
            "Vn": lambda x: f"{x:.0f}",
            },
        alias={
            "S": "power[MVA]",
            "Vn": "voltage_level[kV]",
            },
        virtual={
            "voltage[pu]" : [voltage_pu,["V","Vn"]],
            "voltage[deg]" : [voltage_deg,["V"]],
        },
        include_latlon=True,
        )

    global powerplants_recorder
    powerplants_recorder = geotape.GeoRecorder(gldcore,
        csvname="powerplants.csv",
        objects={"class":"powerplant"},
        properties={
            "status" : status,
            "generator" : generator,
            "fuel" : fuel,
            "S" : lambda x: f"{abs(x):.1f}",
            "operating_capacity": None,
            "total_cost": None,
            },
        alias={
            "S": "power[MVA]",
            "operating_capacity": "capacity[MW]",
            "total_cost": "cost[$]"
            },
        include_latlon=True,            
        )
    return True

def on_commit(t0):
    loads_recorder.sample(t0)
    powerplants_recorder.sample(t0)
    return True