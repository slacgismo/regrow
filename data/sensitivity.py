"""Update weather sensitivity

Syntax: python3 -m sensitivity OPTION ...

Options
-------

    --cooling=DEGC  set the cooling cutoff temperature (default is 10C)
    --freq=FREQ     set the data resample frequency (default is 1H)
    -h|--help|help  generate this help document
    --heating=DEGC  set the heating cutoff temperature (default is 25C)
    --inputs        generate list of input files
    --outputs       generate list of output files
    --update        update only missing/outdates files
    --verbose       verbose progress updates

Description
-----------

The sensivity analysis generates the weather sensitivity for all loads in each
weather location found in the load files.  The nearest weather location is
used to provide solar and temperature data.
"""

import os, sys
import numpy as np
import pandas as pd
import datetime as dt
import cvxpy as cp
import config

INPUTS = {
    "TEMPERATURE" : "geodata/temperature.csv",
    "SOLAR" : "geodata/solar.csv",
    "TOTAL" : "geodata/total.csv",
    # "RESIDENTIAL" : "geodata/residential.csv",
    # "INDUSTRIAL" : "geodata/industrial.csv",
    # "TRANSPORTATION" : "geodata/transportation.csv",
    # "AGRICULTURAL" : "geodata/agricultural.csv",
    }

INTERNAL = {}

OUTPUTS = {
    "BASELOAD" : "geodata/baseload.csv",
    "SENSITIVITY" : "sensitivity.csv",
}
TCOOL = config.Tcool # degC cooling cutoff temperature
THEAT = config.Theat # degC heating cutoff temperature
FREQ = "1h"

from utils import *

options.context = "sensitivity.py"

for arg in read_args(sys.argv,__doc__):
    if arg.startswith("--heating="):
        THEAT = float(arg.split("=")[1])
    elif arg.startswith("--cooling="):
        TCOOL = float(arg.split("=")[1])
    elif arg.startswith("--freq"):
        FREQ = arg.split("=")[1]
    elif arg != "--update":
        raise Exception(f"option '{arg}' is not valid")

def load(KEY):
    verbose(INPUTS[KEY],end="...")
    try:
        result = pd.DataFrame(pd.read_csv(INPUTS[KEY],index_col=[0],parse_dates=[0]).resample(FREQ).mean())
        verbose("ok")
        return result
    except Exception as err:
        verbose(str(err))
        return None

if __name__ == "__main__" and "--update" in sys.argv:

    temperature = load("TEMPERATURE")
    solar = load("SOLAR")
    load = load("TOTAL")
    # TODO: add other inputs

    columns = [f"WD{n:02d}" for n in range(24)]+[f"WE{n:02d}" for n in range(24)]+["TH","TC","S"]
    result = []
    loads = []
    for geohash in load.columns:
        verbose(f"Processing {geohash}",end="...")
        try:

            # data
            T = pd.DataFrame({"T":temperature[nearest(geohash,list(temperature.columns))]})
            S = pd.DataFrame({"S":solar[nearest(geohash,list(solar.columns))]})
            L = pd.DataFrame({"L":load[geohash]})
            data = T.join(S).join(L).dropna()
            T = data["T"].tolist()
            S = data["S"].tolist()
            L = data["L"].tolist()
            t = data.index

            # measurements
            workday = dict([(x,is_workday(x)) for x in t])
            M = [[1 if workday[x] else 0 for x in t]]
            M.extend([[1 if workday[x] and x.hour == h else 0 for x in t] for h in range(1,24)])
            M.append([1 if ~workday[x] else 0 for x in t])
            M.extend([[1 if ~workday[x] and x.hour == h else 0 for x in t] for h in range(1,24)])
            M.extend([
                [x-THEAT if x<THEAT else 0 for x in T], # heating delta T
                [x-TCOOL if x>TCOOL else 0 for x in T], # cooling delta T
                S, # solar load]
            ])
            M = np.array(M).T
            b = np.array(L)
            
            # least squares fit solution
            n = len(M[0])
            m = len(M)
            x = cp.Variable(n)
            def constraint(i):
                result = [0]*n
                result[i] = 1
                return result
            cost = cp.sum_squares(M@x-b) # least squares cost function
            cp.Problem(
                cp.Minimize(cost),
                [ 
                    constraint(0)@x >= 0,
                    constraint(-3)@x <= 0,
                    constraint(-2)@x >= 0,
                ]
                ).solve()
            if x.value is None:
                raise Exception("infeasible")
            else:
                x = x.value

            # ls = {
            #     "Weekday" : [x[0]] + [x[0]+y for y in x[1:24]],
            #     "Weekend" : [x[24]] + [x[24]+y for y in x[25:48]],
            #     }

            Lb = np.array(L) - x[-3:]@M.T[-3:] # baseload 

            verbose("ok")

        except Exception as err:

            Lb = L
            x = [0]*len(columns)
            verbose(err)

        result.append(pd.DataFrame(data=[x[-3:]],index=[geohash],
                                   columns=["heating[MW/degC]","cooling[MW/degC]","solar[MW/W/m^2]"]))
        loads.append(pd.DataFrame(data=Lb,index=t,columns=[geohash]).round(4))

    verbose("Saving output files",end="...")

    result = pd.concat(result).round(3)
    result.index.name = "geocode"
    result.to_csv(OUTPUTS["SENSITIVITY"],index=True,header=True)

    loads = pd.concat(loads,axis=1)
    loads.index.name = "timestamp"
    loads.to_csv(OUTPUTS["BASELOAD"])

    verbose("done")

