"""Update weather sensitivity

Syntax: python3 -m weather_sensitivity OPTIONS ...

Options:
    --cooling=DEGC  set the cooling cutoff temperature (default is 10C)
    --freq=FREQ     set the data resample frequency (default is 1H)
    -h|--help|help  generate this help document
    --heating=DEGC  set the heating cutoff temperature (default is 25C)
    --inputs        generate list of input files
    --outputs       generate list of output files
    --update        update only missing/outdates files
    --verbose       verbose progress updates

The sensivity analysis generates the weather sensitivity for all loads in each
weather location found in the load files.  The nearest weather location is
used to provide solar and temperature data.
"""

import os, sys
import numpy as np
import pandas as pd
import datetime as dt
import geocode as gc

INPUTS = {
    "TEMPERATURE" : "geodata/temperature.csv",
    "SOLAR" : "geodata/solar.csv",
    "COMMERCIAL" : "geodata/commercial.csv",
    "RESIDENTIAL" : "geodata/residential.csv",
    "INDUSTRIAL" : "geodata/industrial.csv",
    "TRANSPORTATION" : "geodata/transportation.csv",
    "AGRICULTURAL" : "geodata/agricultural.csv",
    }

OUTPUTS = {
    "BASELOAD" : "geodata/baseload.csv",
    "SENSITIVITY" : "weather_sensitivity.csv",
}
FORCE = False
VERBOSE = False
TCOOL = 25 # degC cooling cutoff temperature
THEAT = 10 # degC heating cutoff temperature
FREQ = "1h"

if len(sys.argv) == 1:
    print(__doc__)
    exit(1)
elif "--inputs" in sys.argv:
    print(" ".join(INPUTS.values()))
    exit(0)
elif "--outputs" in sys.argv:
    print(" ".join(OUTPUTS.values()))
    exit(0)

for arg in sys.argv[1:]:
    if arg.startswith("--heating="):
        THEAT = float(arg.split("=")[1])
    elif arg.startswith("--cooling="):
        TCOOL = float(arg.split("=")[1])
    elif arg.startswith("--freq"):
        FREQ = arg.split("=")[1]
    elif "--verbose" in sys.argv:
        VERBOSE = True
    elif arg != "--update":
        raise Exception(f"option '{arg}' is not valid")

def verbose(msg,**kwargs):
    if VERBOSE:
        print(msg,file=sys.stderr,flush=True,**kwargs)

if __name__ == "__main__" and "--update" in sys.argv:

    verbose("Loading input files",end="...")
    temperature = pd.DataFrame(pd.read_csv(INPUTS["TEMPERATURE"],index_col=[0],parse_dates=[0]).resample(FREQ).mean())
    solar = pd.DataFrame(pd.read_csv(INPUTS["SOLAR"],index_col=[0],parse_dates=[0]).resample(FREQ).mean())
    load = pd.DataFrame(pd.read_csv(INPUTS["COMMERCIAL"],index_col=[0],parse_dates=[0]).resample(FREQ).mean())
    # TODO: add other inputs
    verbose("done")

    columns = [f"WD{n:02d}" for n in range(24)]+[f"WE{n:02d}" for n in range(24)]+["TH","TC","S"]
    def get_model(geohash):
        _T = pd.DataFrame({"T":temperature[gc.nearest(geohash,temperature.columns)]})
        _S = pd.DataFrame({"S":solar[gc.nearest(geohash,solar.columns)]})
        _L = pd.DataFrame({"L":load[geohash]})
        _L.index = pd.to_datetime(_L.index-dt.timedelta(hours=7),utc=True)
        data = _T.join(_S).join(_L).dropna()
        T = data["T"].tolist()
        S = data["S"].tolist()
        L = data["L"].tolist()
        t = data.index
        
        M = [[1 if x.weekday()<5 else 0 for x in t]]
        M.extend([[1 if x.weekday()<5 and x.hour == h else 0 for x in t] for h in range(1,24)])
        M.append([1 if x.weekday()>4 else 0 for x in t])
        M.extend([[1 if x.weekday()>4 and x.hour == h else 0 for x in t] for h in range(1,24)])
        M.extend([
            [THEAT-x if x<THEAT else 0 for x in T], # heating load
            [TCOOL-x if x>TCOOL else 0 for x in T], # cooling load
            S, # solar load]
        ])
        M= np.array(M).T
        x = np.linalg.solve(M.T@M,M.T@L)
        Y = M@x
        ls = {"Weekday":[x[0]] + [x[0]+y for y in x[1:24]],
              "Weekend":[x[24]] + [x[24]+y for y in x[25:48]],
             }
        return x,M,Y,ls,t,L,T,S

    result = []
    loads = []
    for geohash in load.columns:
        verbose(f"Processing {geohash}",end="...")
        try:
            x,M,Y,ls,t,L,T,S = get_model(geohash)
            a,b,c = x[-3:]
            Lb = np.array(L) - a*np.array(S) - b*np.array([THEAT-x if x<THEAT else 0 for x in T]) - c*np.array([TCOOL-x if x>TCOOL else 0 for x in T])
            verbose("done")
        except Exception as err:
            pd.DataFrame(M,columns=columns).round(4).to_csv(f"weather_sensitivity_{str(err).lower().replace(' ','-')}_{geohash}-M_err.csv")
            Lb = L
            x=[0,0,0]*3
            verbose(err)
        result.append(pd.DataFrame(data=[x[-3:]],index=[geohash],
                                   columns=["heating[MW/degC]","cooling[MW/degC","solar[MW/W/m^2]"]))
        loads.append(pd.DataFrame(data=Lb,index=t,columns=[geohash]).round(4))

    verbose("Saving output files",end="...")
    result = pd.concat(result).round(3)
    result.index.name = "geocode"
    result.to_csv(OUTPUTS["SENSITIVITY"],index=True,header=True)

    loads = pd.concat(loads,axis=1)
    loads.index.name = "timestamp"
    loads.to_csv(OUTPUTS["BASELOAD"])
    verbose("done")

