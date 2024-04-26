"""Update weather sensitivity

Inputs:
- geodata/commercial.csv
- geodata/temperature.csv
- geodata/solar.csv

Outputs:
- geodata/baseload.csv
- weather_sensitivity.csv
"""

import os, sys
import numpy as np
import pandas as pd
import datetime as dt
import geocode as gc

Tcool = 25 # degC cooling cutoff temperature
Theat = 10 # degC heating cutoff temperature

temperature = pd.read_csv("geodata/temperature.csv",index_col=[0],parse_dates=[0])
solar = pd.read_csv("geodata/solar.csv",index_col=[0],parse_dates=[0])
load = pd.DataFrame(pd.read_csv("geodata/commercial.csv",index_col=[0],parse_dates=[0]).resample("1h").mean())

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
        [Theat-x if x<Theat else 0 for x in T], # heating load
        [Tcool-x if x>Tcool else 0 for x in T], # cooling load
        S, # solar load]
    ])
    M= np.array(M).T
    x = np.linalg.solve(M.T@M,M.T@L)
    Y = M@x
    columns = [f"WD{n:02d}" for n in range(24)]+[f"WE{n:02d}" for n in range(24)]+["TH","TC","S"]
    pd.DataFrame(M,columns=columns).round(4).to_csv("M.csv")
    pd.DataFrame([x],columns=columns).round(4).T.to_csv("x.csv",header=False)
    ls = {"Weekday":[x[0]] + [x[0]+y for y in x[1:24]],
          "Weekend":[x[24]] + [x[24]+y for y in x[25:48]],
         }
    return x,M,Y,ls,t,L,T,S

result = []
loads = []
for geohash in load.columns:
    print("Processing",geohash,end="...",flush=True)
    try:
        x,M,Y,ls,t,L,T,S = get_model(geohash)
        a,b,c = x[-3:]
        Lb = np.array(L) - a*np.array(S) - b*np.array([Theat-x if x<Theat else 0 for x in T]) - c*np.array([Tcool-x if x>Tcool else 0 for x in T])
        print("done")
    except Exception as err:
        Lb = L
        x=[0,0,0]*3
        print(err)
    result.append(pd.DataFrame(data=[x[-3:]],index=[geohash],
                               columns=["heating[MW/degC]","cooling[MW/degC","solar[MW/W/m^2]"]))
    loads.append(pd.DataFrame(data=Lb,index=t,columns=[geohash]).round(4))

result = pd.concat(result).round(3)
result.index.name = "geocode"
result.to_csv("weather_sensitivity.csv",index=True,header=True)

loads = pd.concat(loads,axis=1)
loads.index.name = "timestamp"
loads.to_csv(os.path.join("geodata","baseload.csv"))