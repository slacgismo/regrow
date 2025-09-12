"""Generate the load model for each WECC node"""

import os
import load_model_v1 as lm
import pandas as pd
import numpy as np
import time
import datetime as dt
import matplotlib.pyplot as plt
import json

OUTPUT = "WECC240/results" # folder in which results are stored
DATERANGE = pd.date_range("2020-08-01 00:00:00+07:00","2020-09-01 00:00:00+07:00",freq="1h")
REFRESH = True

print("Reading 2018 WECC240 total power",end="...",flush=True)
P = pd.read_csv("../data/geodata/total_2018.csv",index_col=0)
print("ok",flush=True)

print("Reading 2018 WECC240 temperature",end="...",flush=True)
T = pd.read_csv("../data/geodata/temperature_2018.csv",index_col=0)
print("ok",flush=True)

assert (P.columns==T.columns).all(), "columns do not match"

os.makedirs(OUTPUT,exist_ok=True)


columns = [x for x in P.columns if REFRESH or not os.path.exists(f"{OUTPUT}/{x}.txt")]
N = len(columns) # number of nodes to process
tic = time.time() # time processing started

result = []

weather = pd.read_csv("../data/geodata/temperature.csv",index_col=0,parse_dates=[0])*9/5+32
wecc240 = pd.read_csv("../data/geodata/total.csv",index_col=0,parse_dates=[0])
nodes = pd.read_csv("../data/nodes.csv",index_col=0)
wecc240_model = []
wecc240_errors = {}
for n,column in enumerate(columns):
    
    LM_output = f"{OUTPUT}/{column}.txt" # file for results
    LR_plot = f"{OUTPUT}/{column}_LR.png" # file for LR plot
    LM_plot = f"{OUTPUT}/{column}_LM.png" # file for LM plot

    data = pd.DataFrame({"T":T[column]},).join(pd.DataFrame({"P":P[column]}))
    t = data.index
    x = data["T"]
    y = data["P"]

    try:

        title = "{0} ({1})".format(*nodes.loc[column,["Bus  Name","Bus  Number"]].values)
        print("Processing",column,title,f"({n+1}/{N})",end="...",flush=True)

        with open(LM_output,"w") as fh:
            LM = lm.LoadModel((t,x,y),"2018-12",verbose=lambda x:print(str(x),file=fh))

        LM.plot_LR(title=title).savefig(LR_plot)
        LM.plot_LM(title=title).savefig(LM_plot)

        test_data = LM.data[LM.holdout:]

        t = pd.to_datetime(test_data.index).tz_localize("UTC").tz_convert("America/Los_Angeles")
        x = test_data["x"].values
        ya = test_data["y"].values

        new_idx = np.arange(len(LM.data[:LM.holdout]),len(LM.data)) - 1
        PM = lm.Prediction(LM,t=new_idx,x=x)

        yp = PM.y
        pe = (yp/ya-1)*100
        MAPE = np.average(np.ma.MaskedArray(np.abs(pe),mask=np.isnan(pe)))
        MPED = np.std(np.ma.MaskedArray(pe,mask=np.isnan(pe)))
        wecc240_errors[column] = {
            "location":title,
            "new_MAPE":MAPE.round(1),
            "new_MPED":MPED.round(1)}

        fig = PM.plot(t,label="New model prediction")
        fig.plot(t,ya,label="Actual power data")
        fig.plot(wecc240.loc[test_data.index,column],label="Old model prediction")
        fig.legend()
        fig.title(title)
        fig.savefig(os.path.join(OUTPUT,f"{column}_HT.png"))

        label = f"{MAPE=:.1f}% ($3\\sigma={3*MPED:.1f}$%)"

        pe = (wecc240.loc[test_data.index,column].iloc[:-LM.LR.window]/ya-1)*100
        MAPE = np.average(np.ma.MaskedArray(np.abs(pe),mask=np.isnan(pe)))
        MPED = np.std(np.ma.MaskedArray(pe,mask=np.isnan(pe)))
        wecc240_errors[column]["old_MAPE"] = MAPE.round(1)
        wecc240_errors[column]["old_MPED"] = MPED.round(1)

        plt.figure(figsize=(15,10))
        plt.plot(t,pe,label=label)
        plt.xlabel("Date/Time [PST/PDT]")
        plt.ylabel("Holdout error [%]")
        plt.grid()
        plt.legend()
        plt.title(title)
        plt.savefig(os.path.join(OUTPUT,f"{column}_HE.png"))
        plt.close()

        plt.figure(figsize=(15,10))
        plt.hist(pe)
        plt.xlabel("Holdout error [%]")
        plt.ylabel("Occurances")
        plt.grid()
        plt.title(title)
        plt.savefig(os.path.join(OUTPUT,f"{column}_HP.png"))
        plt.close()

        TP = weather[column]
        PW = lm.Prediction(LM,t=range(len(weather.index)),x=TP.values)
        wecc240_model.append(pd.DataFrame(index=weather.index.values,data={column:PW.y}).dropna().round(1))
        pd.concat(wecc240_model,axis=1).to_csv("wecc240_load.csv",index=True,header=True)

        toc = time.time()

        errors = pd.DataFrame(wecc240_errors).T
        errors.index.name = "node"
        errors.to_csv("wecc240_errors.csv",index=True,header=True)

        print("ok",f"(ETA in {dt.timedelta(seconds=round((N/(n+1)-1)*(toc-tic),0))})",flush=True)

    except KeyboardInterrupt as err:

        try:
            os.remove(LR_plot)
        except FileNotFoundError:
            pass
        try:
            os.remove(LM_plot)
        except FileNotFoundError:
            pass
        try:
            os.remove(LM_output)
        except FileNotFoundError:
            pass
