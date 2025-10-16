"""Validate the generation data and cost model"""

from pypower.runopf import runopf
from pypower.ppoption import ppoption
from wecc240 import wecc240
import numpy as np
import pandas as pd
import json

pd.options.display.width = None
pd.options.display.max_columns = None
np.set_printoptions(formatter={'float_kind':"{:12g}".format},linewidth=10000,edgeitems=1000)

# Get loads gridlabd model
glm = json.load(open("../model/wecc240.json","r"))["objects"]

# get busname to busid conversion table
busids = pd.read_csv("bus_data.csv",index_col=[0],dtype=int)
def busid(n:int|float|str):
    return busids.loc[int(n)].busID

# load gen data and gen cost files
gen = pd.read_csv("gen.csv")
gencost = pd.read_csv("gencost.csv")

# load the line data
lines = pd.read_csv("line_data.csv",
    skiprows=1,
    names=['FBUS','TBUS','BR_X','RATE_A','BR_STATUS'],
    )

# assemble bus and branch data
genlist = set([int(x) for x in gen.GEN_BUS])
bus = {}
branch = []
ref = None
for n,line in lines.iterrows():

    # check line data
    if line.FBUS in bus and line.TBUS in bus:
        if line.RATE_A == 99999:
            print(f"WARNING [line_data.csv]: line {n} from {line.FBUS:.0f} to {line.TBUS:.0f} already provided with a usable rating")
        else:
            print(f"WARNING [line_data.csv]: line {n} is a duplicate with different values")
        continue
    elif line.RATE_A == 99999:
        print(f"WARNING [line_data.csv]: line {n} rating 99999 is set to zero for pypower default")
        line.RATE_A = 0.0

    # build bus data
    for BUS_I in [line.FBUS,line.TBUS]:
        if not BUS_I in bus:
            obj = glm[f"wecc240_psse_N_{BUS_I:.0f}"]
            BUS_TYPE = 2 if int(BUS_I) in genlist else 1
            if BUS_TYPE == 2 and ref is None:
                BUS_TYPE = 3
                ref = BUS_I
            PD = float(obj["Pd"].split()[0])
            QD = float(obj["Qd"].split()[0])
            GS = float(obj["Gs"].split()[0])
            BS = float(obj["Bs"].split()[0])
            BUS_AREA = float(obj["area"])
            VM = float(obj["Vm"].split()[0])
            VA = float(obj["Va"].split()[0])
            BASE_KV = float(obj["baseKV"].split()[0])
            ZONE = float(obj["zone"])
            VMAX = float(obj["Vmax"].split()[0])
            VMIN = float(obj["Vmin"].split()[0])
            bus[BUS_I] = [busid(BUS_I),BUS_TYPE,PD,QD,GS,BS,BUS_AREA,VM,VA,BASE_KV,ZONE,VMAX,VMIN]

    # build branch data
    # assert line.RATE_A > 0, f"LINE {n} [{line.FBUS:.0f}-{line.TBUS:.0f}]: zero line ratings"
    branch.append([busid(line.FBUS),busid(line.TBUS),line.BR_X/20,line.BR_X,0.0,line.RATE_A,line.RATE_A,line.RATE_A,0.0,0.0,line.BR_STATUS,-360,+360])

# bus.to_csv("bus.csv")
# branch.to_csv("branch.csv")

pd.DataFrame(bus,index=["BUS_I","BUS_TYPE","PD","QD","GS","BS","BUS_AREA","VM","VA","BASE_KV","ZONE","VMAX","VMIN"]).T.to_csv("bus.csv")
pd.DataFrame(branch,columns=["F_BUS","T_BUS","BR_R","BR_X","BR_B","RATE_A","RATE_B","RATE_C","TAP","SHIFT","BR_STATUS","ANGMIN","ANGMAX"]).to_csv("branch.csv")

# convert gen busname to busid
gen.GEN_BUS = [busid(x) for x in gen.GEN_BUS]

model = {
    "version" : 2,
    "baseMVA" : 100.0,
    "bus" : np.array(sorted(list(bus.values()))),
    "branch" : np.array(sorted(branch)),
    "gen" : gen.to_numpy(),
    "gencost" : gencost.to_numpy(),
}

# # Output pypower case data
# print("from numpy import array")
# print("{")
# for x,y in model.items():
#     if isinstance(y,(int,float)):
#         print(f"    {x}: {y},")
#     else:
#         print(f"    {x}: array(\n      [{',\n       '.join([str(z) for z in y])}\n      ]),")
# print("}")

# run full AC OPF
result = runopf(model,ppoption(VERBOSE=1,OUTALL=3))
print(f"""{result["success"]=}""")
exit(0 if result["success"] else 1)
