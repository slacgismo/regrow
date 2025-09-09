"""Test load model on SCADA data"""

import os
import datetime as dt
import load_model_v1 as lm
import pandas as pd
import numpy as np


CACHE = "test_scada_data.csv.gz"
if CACHE and not os.path.exists(CACHE):
    P=[]

    for year in [2019,2020,2021]:
        file = f"SCADA_data/load_{year}.csv"
        P.append(pd.read_csv(file,
            index_col=0,
            parse_dates=[0],
            ).sum(axis=1))
    P = pd.DataFrame(pd.concat(P,axis=0)).sort_index().clip(lower=0)
    P.columns = ["P"]
    P.index.name = "t"
    P.drop(P[P.P==0].index,inplace=True) # zeros are not valid
    P.dropna(inplace=True) # non-existent DST values are entered as N/A
    P.index = P.index.tz_localize("America/Los_Angeles",ambiguous=[False]*len(P.index))
    P.index = P.index - dt.timedelta(hours=1)

    T = []
    for year in [2019,2020,2021]:
        T.append(pd.read_csv(f"SCADA_data/temp_{year}.csv",
                    index_col=0,
                    parse_dates=[0],
                    ))
    T = pd.DataFrame(pd.concat(T,axis=0)).sort_index()
    T.columns = ["T"]
    T.index.name = "t"
    data = pd.concat([P,T*9/5+32],axis=1).resample("1h").apply(lambda x:x).round(1)
    data.dropna(inplace=True)
    if CACHE:
        data.to_csv(CACHE,
            index=True,
            header=True,
            compression="gzip" if CACHE.endswith(".gz") else None,
            )
    data.index = data.index.astype(str)
else:
    data = pd.read_csv(CACHE,index_col=0)


np.random.seed(42) # what do you get when you multiply nine by six?

LM = lm.LoadModel((data.index,data["T"],data["P"]),holdout="2021",verbose=print)

LM.plot_LR().savefig("test_scada_LR.png")
LM.plot_LM().savefig("test_scada_LM.png")
