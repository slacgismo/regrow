"""WECC 240 model recorders"""

import os
import sys
import datetime as dt

import tape

load_recorder = None

def voltage_pu(data):
    return f"{abs(data['V']) / data['Vn']:.3f}"

def on_init(t0):
    global load_recorder
    load_recorder = tape.GeoRecorder(gridlabd,
        csvname="loads.csv",
        objnames="wecc240_psse_L",
        properties={
            "status" : None,
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
    return True

def on_commit(t0):
    load_recorder.sample(t0)
    return True