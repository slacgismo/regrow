import os
import load_model_v1 as lm
import pandas as pd
import time
import datetime as dt

print("Reading total power",end="...",flush=True)
P = pd.read_csv("../data/geodata/total_2018.csv",index_col=0)
print("ok",flush=True)

print("Reading temperature",end="...",flush=True)
T = pd.read_csv("../data/geodata/temperature_2018.csv",index_col=0)
print("ok",flush=True)

assert (P.columns==T.columns).all(), "columns do not match"

model={}
N = len(P.columns)
tic = time.time()
PATH="results"
os.makedirs(PATH,exist_ok=True)
for n,column in enumerate(P.columns):
    
    LM_output = f"{PATH}/{column}.txt"
    LR_plot = f"{PATH}/{column}_LR.png"
    LM_plot = f"{PATH}/{column}_LM.png"
    if os.path.exists(LR_plot) and os.path.exists(LM_plot):
        continue

    print("Processing",column,f"({n+1}/{N})",end="...",flush=True)
    data = pd.DataFrame({"T":T[column]},).join(pd.DataFrame({"P":P[column]}))
    
    t = data.index
    x = data["T"]
    y = data["P"]

    try:
        with open(LM_output,"w") as fh:
            LM = lm.LoadModel((t,x,y),"2018-12",verbose=lambda x:print(str(x),file=fh))
        LM.plot_LR().savefig(LR_plot)
        LM.plot_LM().savefig(LM_plot)
    except KeyboardInterrupt as err:
        os.remove(LR_plot)
        os.remove(LM_plot)
        os.remove(LM_output)
    toc = time.time()
    print("ok",f"(ETA in {dt.timedelta(seconds=round((N/(n+1)-1)*(toc-tic),0))})",flush=True)