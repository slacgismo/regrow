import os
import time
import numpy as np
import pandas as pd
import load_model_v1 as lm

pd.options.display.max_columns=None
pd.options.display.width=None
pd.options.display.max_rows=None

FOLDER = "AMI_data"
POWER = os.path.join(FOLDER,"load.csv")
TEMPERATURE = os.path.join(FOLDER,"temperature.csv")
DATA = os.path.join(FOLDER,"data.csv")
OUTPUT = os.path.join(FOLDER,"results")
os.makedirs(OUTPUT,exist_ok=True)

np.random.seed(42) # what do you get when you multiply nine by six?

if not os.path.exists(DATA):
    #
    # Read load data
    #
    P = pd.read_csv(POWER)
    Pt = [f"{x.record_date} {x.hour_id-1:02d}:00:00{-x.utc_offset:+03d}:00" for n,x in P.iterrows()]
    P.index = pd.DatetimeIndex(pd.to_datetime(Pt,utc=True))
    P.drop(["record_date","hour_id","utc_offset"],axis=1,inplace=True)
    P.columns = ["P[MW]"]
    P["P[MW]"] = (P["P[MW]"]/1000).round(1)

    #
    # Read temperature data
    #
    T = pd.read_csv(TEMPERATURE,index_col=0,parse_dates=[0])
    T.columns = ["T[degF]"]
    T.join(P).drop_na(),to_csv(DATA,index=True,header=True)

df = pd.read_csv(DATA,index_col=0)
t = df.index
x = df["T[degF]"].values
y = df["P[MW]"].values

#
# Generate load model
#
with open(os.path.join(OUTPUT,"ami.txt"),"w") as fh:
    LM = lm.LoadModel((t,x,y),"2015-12",window=3,verbose=lambda x:print(x,file=fh))
LM.plot_LR().savefig(os.path.join(OUTPUT,"ami_LR.png"))
LM.plot_LM().savefig(os.path.join(OUTPUT,"ami_LM.png"))
