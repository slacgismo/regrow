import marimo

__generated_with = "0.3.9"
app = marimo.App(width="full")


@app.cell
def __():
    #
    # This notebook is used to test and validate the sensitivity analysis module
    #
    Tcool = 25 # degC cooling cutoff temperature
    Theat = 10 # degC heating cutoff temperature
    return Tcool, Theat


@app.cell
def __(pd):
    #
    # Load data
    #
    temperature = pd.read_csv("geodata/temperature.csv",index_col=[0],parse_dates=[0])
    solar = pd.read_csv("geodata/solar.csv",index_col=[0],parse_dates=[0])
    load = pd.DataFrame(pd.read_csv("geodata/commercial.csv",index_col=[0],parse_dates=[0]).resample("1h").mean())
    return load, solar, temperature


@app.cell
def __(Tcool, Theat, dt, err, load, np, os, pd, solar, temperature):
    #
    # Construct model
    #
    import geocode as gc
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
        loads.append(pd.DataFrame(data=Lb,index=t,columns=[geohash]))

    result = pd.concat(result).round(3)
    result.index.name = "geocode"
    result.to_csv("weather_sensitivity.csv",index=True,header=True)

    loads = pd.concat(loads)
    loads.index.name = "timestamp"
    loads.to_csv(os.path.join("geodata","baseload.csv"))
    return (
        L,
        Lb,
        M,
        S,
        T,
        Y,
        a,
        b,
        c,
        gc,
        geohash,
        get_model,
        loads,
        ls,
        result,
        t,
        x,
    )


@app.cell
def __(load, ls, pl):
    pl.plot(ls["Weekday"],label='Weekday')
    pl.plot(ls["Weekend"],label='Weekday')
    pl.grid()
    pl.title(f"Base loadshape for commercial buildings at node {load.columns[0]}")
    pl.xlabel('UTC hour')
    pl.ylabel('MW')
    pl.legend()
    return


@app.cell
def __(L, Y, np, pl, t):
    pl.figure(figsize=(20,10))
    pl.plot(t,Y,label='model')
    pl.plot(t,L,label='data')
    pl.ylabel('MW')
    pl.grid()
    pl.title(f"MAPE ={np.mean(100*(L-Y)/L).round(2)}%")
    pl.legend()
    return


@app.cell
def __(L, Y, pl):
    pl.figure(figsize=(20,10))
    pl.hist(bins=range(0,100,1),x=100*(Y-L)/L,label='Model error',density=True)
    pl.ylabel('Probability density')
    pl.xlabel('% error')
    pl.grid()
    pl.legend()
    return


@app.cell
def __():
    import os, sys
    import marimo as mo
    import numpy as np
    import scipy as sp
    import pandas as pd
    import datetime as dt
    import matplotlib.pyplot as pl
    return dt, mo, np, os, pd, pl, sp, sys


@app.cell
def __():
    # Check weather data

    # weather = pd.read_csv("nsrdb/9mtzm4.csv.zip",index_col=[0],parse_dates=[0],low_memory=False)
    # weather.groupby(weather.index.date).sum().plot(y="dni",figsize=(20,10))
    # weather.groupby(weather.index.date).mean().plot(y="temp_air",figsize=(20,10))
    # pl.axhline(37, ls='--', color='orange', linewidth=1)
    return


if __name__ == "__main__":
    app.run()
