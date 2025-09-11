import os
import time
import numpy as np
import pandas as pd
import load_model_v1 as lm
import matplotlib.pyplot as plt

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
holdout = "2015-12"
with open(os.path.join(OUTPUT,"ami.txt"),"w") as fh:
    LM = lm.LoadModel((t,x,y),holdout=holdout,window=3,verbose=lambda x:print(x,file=fh))
LM.plot_LR().savefig(os.path.join(OUTPUT,"ami_LR.png"))
LM.plot_LM().savefig(os.path.join(OUTPUT,"ami_LM.png"))

#
# Generate holdout prediction test
#
test_data = LM.data[holdout:]

t = pd.to_datetime(test_data.index).tz_convert("America/Los_Angeles")
x = test_data["x"].values
ya = test_data["y"].values

new_idx = np.arange(len(LM.data[:holdout]),len(LM.data)) - 1
PM = lm.Prediction(LM,t=new_idx,x=x)

yp = PM.y
pe = (yp/ya-1)*100
MAPE = round(float(np.average(np.ma.MaskedArray(pe,mask=np.isnan(pe)))),1)

fig = PM.plot(t,label="Predicted power")
fig.plot(t,ya,label="Actual power")
fig.legend()
fig.title("AMI holdout test")
fig.savefig(os.path.join(OUTPUT,"ami_HT.png"))

plt.figure(figsize=(15,10))
plt.plot(t,pe,label=f"{MAPE=}%")
plt.xlabel("Date/Time [PST/PDT]")
plt.ylabel("Holdout error [%]")
plt.grid()
plt.legend()
plt.title("AMI holdout test")
plt.savefig(os.path.join(OUTPUT,"ami_HE.png"))

plt.figure(figsize=(15,10))
plt.hist(pe,label=f"{MAPE=}%")
plt.xlabel("Holdout error [%]")
plt.ylabel("Occurances")
plt.grid()
plt.legend()
plt.title("AMI holdout test")
plt.savefig(os.path.join(OUTPUT,"ami_HP.png"))
