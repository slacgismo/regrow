"""WECC 240 model recorders"""

import os
import sys
import datetime as dt

import tape

loads_recorder = None
plant_recorder = None

def voltage_pu(data):
    return f"{abs(data['V']) / data['Vn']:.3f}"

def on_init(t0):

    global loads_recorder
    loads_recorder = tape.GeoRecorder(gridlabd,
        csvname="loads.csv",
        objects=r"^wecc240_psse_L_[0-9]+",
        properties={
            "status" : lambda x:["OFFLINE","ONLINE","CURTAILED"][x],
            "S" : lambda x: f"{abs(x):.1f}",
            },
        alias={
            "S": "power[MVA]",
            },
        virtual={
            "voltage[pu]" : [voltage_pu,["V","Vn"]],
        },
        include_latlon=True,
        )

    global powerplants_recorder
    powerplants_recorder = tape.GeoRecorder(gridlabd,
        csvname="powerplants.csv",
        objects={"class":"powerplant"},
        properties={
            "status" : lambda x:["OFFLINE","ONLINE","CURTAILED"][x],
            "S" : lambda x: f"{abs(x):.1f}",
            },
        alias={
            "S": "power[MVA]",
            },
        include_latlon=True,            
        )
    return True

def on_commit(t0):
    loads_recorder.sample(t0)
    powerplants_recorder.sample(t0)
    return True