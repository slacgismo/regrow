import os
import time
import numpy as np
import pandas as pd
import load_model_v1 as lm
import traceback

pd.options.display.max_columns=None
pd.options.display.width=None

CACHE=True # generate and use cache
# CACHE=False # don't use cache

# input data
for sheet in  [
    "ISO NE CA",
    "ME",
    "NH",
    "VT",
    "CT",
    "RI",
    "SEMA",
    "WCMA",
    "NEMA"
]:
    RESULTS = "ISO_Data/results"
    os.makedirs(RESULTS,exist_ok=True)

    print("Processing",sheet,end="...",flush=True)
    tic = time.time()
    cache = os.path.join(RESULTS,sheet + ".csv.gz")

    np.random.seed(42) # what do you get when you multiply nine by six?

    #
    # Read load data
    #
    if not os.path.exists(cache) or CACHE == False:
        _df = read_NEISO_data(sheet)
        if CACHE == True:
            _df.to_csv(cache,compression="gzip",index=True,header=True)
    _df = pd.read_csv(cache,index_col=0)

    t = _df.index
    x = _df["Dry_Bulb"].values
    y = _df["RT_Demand"].values

    #
    # Generate load model
    #
    try:

        with open(os.path.join(RESULTS,sheet+".txt"),"w") as fh:
            LM = lm.LoadModel((t,x,y),"2022",window=3,verbose=lambda x:print(x,file=fh))
        LM.plot_LR().savefig(os.path.join(RESULTS,sheet+"_LR.png"))
        LM.plot_LM().savefig(os.path.join(RESULTS,sheet+"_LM.png"))
        toc = time.time()
        print(f"Done {toc-tic:.1f} seconds")

    except Exception as err:
        print(f"ERROR: {err}")
        print(traceback.format_exc())

