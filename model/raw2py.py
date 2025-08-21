"""Convert PSSE "raw" file to PyPOWER "py" file"""

# Copyright (C) 2025, Eudoxys Sciences LLC
# Author: dchassin@eudoxys.com
# Project: REGROW (NREL-SUB-2025-10301)

import os
import sys
import numpy as np
import pandas as pd

class PSSEraw:

    def __init__(self,filename):
        with open(filename,"r") as fh:
            psse = fh.readlines()

        self.name = os.path.splitext(os.path.basename(filename))[0]

        if os.path.exists(self.name+"_cost.csv"):
            try:
                self.gencost = pd.read_csv(self.name+"_cost.csv",index_col=[0])
                self.gencost = self.gencost.to_dict('index')
            except pd.errors.EmptyDataError:
                os.remove(self.name+"_cost.csv")
                self.gencost = {}
        else:
            self.gencost = {}

        section = 'SYSTEM-WIDE DATA'
        data = {section:[]}

        def convert(x):
            try:
                return int(x)
            except:
                pass
            try:
                return float(x)
            except:
                pass
            return str(x.strip("'").strip())

        for line in psse:
            if line[0] == '0':
                if not "BEGIN " in line:
                    break
                section = line.split("BEGIN ")[1].strip()
                data[section] = []
            elif line[0] == '@':
                pass
            elif line[0] == ' ':
                data[section].append([convert(x.strip()) for x in line.split(",")])
        self.section = data
        self.mvabase = self.section["SYSTEM-WIDE DATA"][0][1]
        self.version = self.section["SYSTEM-WIDE DATA"][0][2]
        assert self.version >= 32, "PSS/E versions older than 32 are not supported"

    def to_pypower(self,file=None):

        case = {
            "version" : "2",
            "baseMVA" : self.mvabase,
            "bus" : [],
            "gen" : [],
            "branch": [],
            "gencost": [],
            "dcline": [],
            "dclinecost": [],
        }

        done = {"SYSTEM-WIDE DATA":"ok"}

        # busses
        bus_index = {}
        bus_map = {}
        for n,m in enumerate(self.section["BUS DATA"]):
            bus_i = m[0]
            bus_index[n] = bus_i
            bus_map[m[0]] = n
        done["BUS DATA"] = "ok"

        # loads
        Pd = {}
        Qd = {}
        for n,m in enumerate(self.section["LOAD DATA"]):
            bus_i = m[0]
            if bus_i not in Pd:
                Pd[bus_i] = np.array([0.0,0.0,0.0])
                Qd[bus_i] = np.array([0.0,0.0,0.0])
            Pd[bus_i] += [m[5],m[7],m[9]] if m[2] else [0.0,0.0,0.0]
            Qd[bus_i] += [m[6],m[8],m[10]] if m[2] else [0.0,0.0,0.0]
        done["LOAD DATA"] = "ok"

        # fixed shunts
        Gs = {}
        Bs = {}
        for n,m in enumerate(self.section["FIXED SHUNT DATA"]):
            bus_i = m[0]
            if bus_i not in Gs:
                Gs[bus_i] = 0.0
                Bs[bus_i] = 0.0
            Gs[bus_i] += m[3] if m[2] else 0.0
            Bs[bus_i] += m[4] if m[2] else 0.0
        done["FIXED SHUNT DATA"] = "ok"

        # switched shunts
        for n,m in enumerate(self.section["SWITCHED SHUNT DATA"]):
            bus_i = m[0]
            if bus_i not in Gs:
                Bs[bus_i] = 0.0
            Bs[bus_i] += m[9] if m[3] else 0.0
        done["SWITCHED SHUNT DATA"] = "ok"

        # branches
        for n,m in enumerate(self.section["BRANCH DATA"]):
            fbus,tbus = bus_map[m[0]],bus_map[abs(m[1])]
            r,x,b = m[3],m[4],m[5]
            rateA,rateB,rateC = m[7],m[8],m[9]
            status = m[14]
            case["branch"].append([fbus,tbus,r,x,b,rateA,rateB,rateC,0.0,0.0,status,-360.0,360.0])
            if status:
                gi,bi,gj,bj = m[19],m[20],m[21],m[22]
                if fbus not in Gs:
                    Gs[fbus] = 0.0
                    Bs[fbus] = 0.0
                if tbus not in Gs:
                    Gs[tbus] = 0.0
                    Bs[tbus] = 0.0
                Gs[fbus] += gi * self.mvabase 
                Bs[fbus] += bi * self.mvabase 
                Gs[tbus] += gj * self.mvabase 
                Bs[tbus] += bj * self.mvabase 
        done["BRANCH DATA"] = "ok"

        # busses
        for n,m in enumerate(self.section["BUS DATA"]):
            bus_i = m[0]
            vm = m[9]
            p = Pd[bus_i] if bus_i in Pd else [0.0,0.0,0.0]
            q = Qd[bus_i] if bus_i in Qd else [0.0,0.0,0.0]
            g = Gs[bus_i] if bus_i in Gs else 0.0
            b = Bs[bus_i] if bus_i in Bs else 0.0
            case["bus"].append([n,m[3],round(p[0]+(p[1]+p[2]*vm)*vm,1),round(q[0]+(q[1]+q[2]*vm)*vm,1),g,b,m[4],m[7],m[8],m[2],m[5],m[9],m[10]])
        done["BUS DATA"] = "ok"

        # generators
        for n,m in enumerate(self.section["GENERATOR DATA"]):
            bus_i = bus_map[m[0]]
            case["gen"].append([bus_i,m[2],m[3],m[4],m[5],m[6],m[8],m[14],m[16],m[17]])
            busname = self.section["BUS DATA"][bus_i][1]
            genname = f"{busname}_{m[1]}"
            try:
                p = [self.gencost[genname][x] for x in ["model","startup","shutdown","cost"]]
            except KeyError:
                p = [2,0,0,"0,0,0"]
                self.gencost[genname] = p                
            d = [float(x) for x in p[3].split(",")]
            gencost = [p[0],p[1],p[2],len(d)] + d
            case["gencost"].append(gencost)
        done["GENERATOR DATA"] = "ok"

        # dclines

        # dcline costs

        # write case file
        with open(file if file else self.name+".py","w") as fh:
            print(f"from numpy import array",file=fh)
            print(f"def {self.name}():",file=fh)
            print(f"   return {{",file=fh)
            for tag,data in case.items():
                if isinstance(data,list):
                    print(f"""    "{tag}": array([""",file=fh)
                    for row in data:
                        print(f"""        {row},""",file=fh)
                    print("    ]),",file=fh)
                else:
                    print(f"""    "{tag}": {repr(data)},""",file=fh)
            print(f"}}",file=fh)

        # write cost file
        if not os.path.exists(self.name+"_cost.csv"):
            df = pd.DataFrame(self.gencost).T
            df.columns = ["model","startup","shutdown","cost"]
            df.index.name = "id"
            df.to_csv(self.name+"_cost.csv",header=True,index=True)

        return done

model = PSSEraw("wecc240_psse.raw")
done = model.to_pypower()
print(f"{model.name} Summary")
print(f"{'-'*len(model.name)}--------")
for name,data in model.section.items():
    if len(data) > 0:
        print(f"  {name.title()}{'.'*(30-len(name))} {len(data):4d} item{'s' if len(data)>1 else ' '} ({done[name] if name in done else 'not processed'})")
