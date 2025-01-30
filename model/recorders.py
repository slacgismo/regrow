"""WECC 240 model recorders"""

import os
import sys
import datetime as dt

import tape

load_recorder = None

def on_init(t0):
    global load_recorder
    load_recorder = tape.GeoRecorder(gridlabd,"loads.csv","wecc240_psse_L",{
        "status" : None,
        "V" : lambda x: f"{abs(x):.3f}",
        "Vn": lambda x: f"{x:.3f}",
        "S" : lambda x: f"{abs(x):.1f}"
        })
    return True

def on_commit(t0):
    load_recorder.sample(t0)
    return True