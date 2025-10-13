"""Convert generation data to PyPOWER gen and gencost arrays

Run this script to generate the gencost.csv file.

Call model("generation_cost) to obtain the pypower gen and gencost data.
"""

import sys
import numpy as np
import pandas as pd

import pypower.idx_bus as bus
import pypower.idx_brch as branch
import pypower.idx_gen as gen
import pypower.idx_cost as cost

pp_index = {

    "bus": {getattr(bus,x):x for x in dir(bus) if x[0] != "_" and x not in {"PQ","PV","REF","NONE"}},
    "branch": {getattr(branch,x):x for x in dir(branch) if x[0] != "_"},
    "gen": {getattr(gen,x):x for x in dir(gen) if x[0] != "_"},
    "gencost": {getattr(cost,x):x for x in dir(cost) if x[0] != "_"},
}

def gencost(
        csvfile:str,
        *,
        maxorder:int=2,
        ) -> np.ndarray:
    """Generation cost data

    csvfile: name of CSV file from which to read generation data

    maxorder: maximum polynomial model order to generate (default 2)
    """
    gen = pd.read_csv(csvfile)

    gencost = []
    for n,data in gen.iterrows():

        # read price data
        prices = [(data[f"MW{n+1}"], data[f"Cost{n+1}"])
                for n in range(4)
                if n == 0 or ( f"Cost{n+1}" in gen.columns and data[f"Cost{n+1}"] > 0 and data[f"Cost{n}"] < data[f"Cost{n+1}"] )
            ]
        Q,P = zip(*prices)

        x = []
        y = []
        for m in range(len(P)):
            x.append(np.arange(Q[m-1] if m>0 else 0,Q[m]+1,1).round(0))
            y.append(np.ones(len(x[-1]))*P[m])
        x = np.hstack(x)
        y = np.cumsum(np.hstack(y)) + data.No_Load_Cost
        if x[-1] < data["Pmax"]:
            print(f"WARNING [{data['genname']}@{n}]: non-convex prices from {x[-1]:.1f} to {data['Pmax']:.1f} MW relaxed from $0.00/MWh to ${prices[-1][1]:.2f}/MWh")
            x[-1] = data["Pmax"]

        model = 2 # polynomial
        startup = data["SUCost"]
        shutdown = data["SDCost"]
        k = len(P)
        p = np.polyfit(x,y,min(k-1,maxorder)).tolist()
        p[-1] += data.No_Load_Cost
        gencost.append([model,float(startup),float(shutdown),len(p)] + p + [0]*(maxorder-k+1))
    
    return np.array(gencost)

def gen(
        csvfile:str,
        *,
        Qfactor:float=0.0,
        VG:float=1.0,
        MBASE:float=100.0,
    ) -> np.ndarray:
    """Generation unit data

    csvfile: name of CSV file from which to read generation data

    Qfactor: fraction of real power that can be used to generate reactive
             power (default 0.2)

    VG: voltage magnitude setpoint (default 1.0)

    MBASE: total MVA base of machine (default 100)
    """
    gen = pd.read_csv(csvfile)
    N = len(gen)
    return np.array([
        gen.busname, # GEN_BUS
        gen.InitPow, # PG
        np.zeros(N), # QG
        gen.Pmax*np.full(N,Qfactor), # QMAX
        -gen.Pmax*np.full(N,Qfactor), # QMIN
        np.full(N,VG), # VG
        np.full(N,MBASE), # MBASE
        gen.InitStatus, # GEN_STATUS
        gen.Pmax, # PMAX
        gen.Pmin, # PMIN
        np.zeros(N), # PC1
        np.zeros(N), # PC2
        np.zeros(N), # PC1
        np.zeros(N), # QC1MIN
        np.zeros(N), # QC1MAX
        np.zeros(N), # QC2MIN
        np.zeros(N), # QC2MAX
        gen.Ramp_Rate, # RAMP_AGC
        np.zeros(N), # RAMP_10
        np.zeros(N), # RAMP_30
        np.zeros(N), # RAMP_Q
        np.zeros(N), # APF
        ]).T

def model(
        csvfile:str,
        *,
        maxorder:int=2,
        bus:np.array=None,
        fail=lambda x: print(f"bus {x} not found",file=sys.stderr)
        ) -> dict:
    """Generation cost data

    csvfile: name of CSV file from which to read generation data

    maxorder: maximum polynomial model order to generate (default 2)

    """
    result = {"gen":gen(csvfile),"gencost":gencost(csvfile,maxorder=maxorder)}
    if not bus is None:
        bus_i = set(bus[0,:])
        for gen_bus in result["gen"][1,:]:
            if gen_bus not in bus_i:
                fail(gen_bus)
    return result

if __name__ == "__main__":

    def fix(a,n=0):
        """Fill blank trailing columns names with numbered columns based on
        last named column
        """
        if not a[-1]:
            m,p,n = fix(a[:-1],n)
            return m + [f"{p}{n}"],p,n+1
        else:
            return a[:-1] + [f"{a[-1]}{n}"],a[-1],n+1

    for key,data in model("generation_data.csv").items():
        with open(f"{key}.csv","w") as fh:
            if key in pp_index:
                header = fix([pp_index[key][n]  if n < len(pp_index[key]) else "" for n in range(len(data[0]))])[0]
                print(header,file=fh)
                print(",".join(header),file=fh)
                values = [",".join([f"{y:g}" for y in x]) for x in data.tolist() + [0.0]*(len(pp_index[key])-len(data))]
                print(*values,sep="\n",file=fh)
                print(file=fh)
