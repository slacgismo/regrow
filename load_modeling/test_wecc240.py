"""Generate the load model for each WECC node"""

import os
import load_model_v1 as lm
import pandas as pd
import time
import datetime as dt

OUTPUTPATH = "WECC240/results" # folder in which results are stored
DATERANGE = pd.date_range("2020-08-01 00:00:00+07:00","2020-09-01 00:00:00+07:00",freq="1h")
REFRESH = True

print("Reading 2018 WECC240 total power",end="...",flush=True)
P = pd.read_csv("../data/geodata/total_2018.csv",index_col=0)
print("ok",flush=True)

print("Reading 2018 WECC240 temperature",end="...",flush=True)
T = pd.read_csv("../data/geodata/temperature_2018.csv",index_col=0)
print("ok",flush=True)

assert (P.columns==T.columns).all(), "columns do not match"

os.makedirs(OUTPUTPATH,exist_ok=True)


columns = [x for x in P.columns if REFRESH or not os.path.exists(f"{OUTPUTPATH}/{x}.txt")]
N = len(columns) # number of nodes to process
tic = time.time() # time processing started

result = []

weather = pd.read_csv("../data/geodata/temperature.csv",index_col=0,parse_dates=[0])
# print(weather)

for n,column in enumerate(columns):
    
    LM_output = f"{OUTPUTPATH}/{column}.txt" # file for results
    LR_plot = f"{OUTPUTPATH}/{column}_LR.png" # file for LR plot
    LM_plot = f"{OUTPUTPATH}/{column}_LM.png" # file for LM plot

    data = pd.DataFrame({"T":T[column]},).join(pd.DataFrame({"P":P[column]}))
    t = data.index
    x = data["T"]
    y = data["P"]

    try:

        print("Processing",column,f"({n+1}/{N})",end="...",flush=True)

        with open(LM_output,"w") as fh:
            LM = lm.LoadModel((t,x,y),"2018-12",verbose=lambda x:print(str(x),file=fh))
        LM.plot_LR().savefig(LR_plot)
        LM.plot_LM().savefig(LM_plot)

        toc = time.time()

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

    # ndx = (DATERANGE - pd.to_datetime(t[0]+"+00:00"))
    # tt = ndx.days*24 + ndx.seconds/3600
    # xt = weather.loc[pd.DatetimeIndex([t[0]]*len(tt))+ndx][column].values
    # yt = LM.predict_load((tt,xt))

    # data = pd.DataFrame(data={"T":xt,"P":yt},index=tt)
    # data.to_csv(f"{OUTPUTPATH}/{column}.csv",index=True,header=True)
    # result.append(pd.DataFrame(data={column:yt},index=tt))


# pd.concat(result,axis=1).to_csv("wecc240_load.csv",index=True,header=True)
