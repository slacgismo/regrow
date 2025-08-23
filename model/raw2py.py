"""Convert PSSE "raw" file to PyPOWER "py" file"""

# Copyright (C) 2025, Eudoxys Sciences LLC
# Author: dchassin@eudoxys.com
# Project: REGROW (NREL-SUB-2025-10301)

import os
import sys
import json
import importlib
import numpy as np
import pandas as pd

import pypower.idx_bus as pp_bus
import pypower.idx_brch as pp_branch
import pypower.idx_gen as pp_gen
import pypower.idx_dcline as pp_dcline
from pypower.runpf import runpf
from pypower.runopf import runopf

import psse
    
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

        self.headings = dict(data)
        for line in psse:
            if line.startswith("0 / "): # section delimiter
                if not "BEGIN " in line:
                    break
                section = line.split("BEGIN ")[1].strip()
                data[section] = []
                self.headings[section] = []
            elif line.startswith("@!"): # section metdata
                self.headings[section].extend(line[2:].strip().split(','))
            else:
                values = [convert(x.strip()) for x in line.split(",")]
                if len(data[section]) == 0 or len(data[section][-1])+len(values) >= len(self.headings[section]):
                    data[section].append(values)
                else:
                    data[section][-1].extend(values)
        self.section = data
        self.mvabase = self.section["SYSTEM-WIDE DATA"][0][1]
        self.version = self.section["SYSTEM-WIDE DATA"][0][2]
        assert self.version >= 32, "PSS/E versions older than 32 are not supported"

        # check data rows for correct number of items
        for section,data in self.section.items():
            p = len(self.headings[section])
            if p == 0 or section == "SYSTEM-WIDE DATA":
                continue
            for n,m in enumerate(data):
                if len(self.headings[section]) < len(m):
                    print(f"WARNING [{section}]: row {n} incorrect length (expected {p} items, found {len(m)})")

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
        bus_name = {}
        for n,m in enumerate(self.section["BUS DATA"]):
            bus_i = m[psse.bus.I]
            bus_index[n] = bus_i
            bus_map[bus_i] = n
            bus_name[n] = m[psse.bus.NAME]
        done["BUS DATA"] = "ok"

        # loads
        Pd = {}
        Qd = {}
        for n,m in enumerate(self.section["LOAD DATA"]):
            bus_i = m[psse.bus.I]
            if bus_i not in Pd:
                Pd[bus_i] = np.array([0.0,0.0,0.0])
                Qd[bus_i] = np.array([0.0,0.0,0.0])
            Pd[bus_i] += [m[psse.load.PL],m[psse.load.IP],m[psse.load.YP]] if m[psse.load.STAT] else [0.0,0.0,0.0]
            Qd[bus_i] += [m[psse.load.QL],m[psse.load.IQ],m[psse.load.YQ]] if m[psse.load.STAT] else [0.0,0.0,0.0]
        done["LOAD DATA"] = "ok"

        # fixed shunts
        Gs = {}
        Bs = {}
        for n,m in enumerate(self.section["FIXED SHUNT DATA"]):
            bus_i = m[psse.bus.I]
            if bus_i not in Gs:
                Gs[bus_i] = 0.0
                Bs[bus_i] = 0.0
            Gs[bus_i] += m[psse.fixed_shunt.GL] if m[psse.fixed_shunt.STATUS] else 0.0
            Bs[bus_i] += m[psse_fixed_shunt.BL] if m[psse.fixed_shunt.STATUS] else 0.0
        done["FIXED SHUNT DATA"] = "ok"

        # switched shunts
        for n,m in enumerate(self.section["SWITCHED SHUNT DATA"]):
            bus_i = m[psse.bus.I]
            if bus_i not in Gs:
                Bs[bus_i] = 0.0
            Bs[bus_i] += m[psse.switched_shunt.BINIT] if m[psse.switched_shunt.ST] else 0.0
        done["SWITCHED SHUNT DATA"] = "ok"

        # branches
        for n,m in enumerate(self.section["BRANCH DATA"]):
            fbus,tbus = bus_map[m[psse.branch.I]],bus_map[abs(m[psse.branch.J])] # negative J means metered branch
            r,x,b = m[psse.branch.R],m[psse.branch.X],m[psse.branch.B]
            rateA,rateB,rateC = m[psse.branch.RATE1],m[psse.branch.RATE2],m[psse.branch.RATE3]
            status = m[psse.branch.STAT]
            case["branch"].append([fbus+1,tbus+1,round(r,6),round(x,6),round(b,6),rateA,rateB,rateC,0.0,0.0,status,-360.0,360.0])
            if status: # include branch shunt in from and to busses
                gi,bi,gj,bj = m[psse.branch.GI],m[psse.branch.BI],m[psse.branch.GJ],m[psse.branch.BJ]
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

        # transformers
        Zbase = (np.array([x[psse.bus.BASEKV] for x in self.section["BUS DATA"]])**2 / self.mvabase).tolist()
        for n,m in enumerate(self.section["TRANSFORMER DATA"]):
            if m[psse.transformer.K] > 0:
                print(f"WARNING: three-winding transformer {n} is not supported")
                continue
            fbus = bus_map[m[psse.transformer.I]]
            tbus = bus_map[m[psse.transformer.J]]
            Zbf,Zbt = Zbase[fbus],Zbase[tbus]
            r = m[psse.transformer.R12] / Zbf
            x = m[psse.transformer.X12] / Zbf
            rateA,rateB,rateC = m[psse.transformer.RATE11:psse.transformer.RATE14]
            status = m[psse.transformer.STAT]
            case["branch"].append([fbus+1,tbus+1,round(r,6),round(x,6),round(b,6),rateA,rateB,rateC,1.0,0.0,status,-360.0,360.0])
        done["TRANSFORMER DATA"] = "ok"

        # busses
        for n,m in enumerate(self.section["BUS DATA"]):
            bus_i = m[psse.bus.I]
            vm = m[psse.bus.VM]
            p = Pd[bus_i] if bus_i in Pd else [0.0,0.0,0.0]
            q = Qd[bus_i] if bus_i in Qd else [0.0,0.0,0.0]
            g = Gs[bus_i] if bus_i in Gs else 0.0
            b = Bs[bus_i] if bus_i in Bs else 0.0
            case["bus"].append([n+1,m[psse.bus.BUSTYPE],round(p[0]+(p[1]+p[2]*vm)*vm,1),round(q[0]+(q[1]+q[2]*vm)*vm,1),g,b,m[4],m[7],m[8],m[2],m[5],m[9],m[10]])
        done["BUS DATA"] = "ok"

        # generators
        for n,m in enumerate(self.section["GENERATOR DATA"]):
            bus_i = bus_map[m[psse.gen.I]]
            case["gen"].append([bus_i+1,m[psse.gen.PG],m[psse.gen.QG],m[psse.gen.QT],
                m[psse.gen.QB],m[psse.gen.VS],m[psse.gen.MBASE],m[psse.gen.STAT],
                m[psse.gen.PT],m[psse.gen.PB]])
            genname = f"{self.section['BUS DATA'][bus_i][psse.bus.NAME]}_{m[psse.gen.ID]}"

            # gencost
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
        for n,m in enumerate(self.section["TWO-TERMINAL DC DATA"]):
            case["dcline"].append(m)
        done["TWO-TERMINAL DC DATA"] = "TODO"

        # dcline costs
        # TODO

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

        # save bus map
        busndx = pd.DataFrame({ndx:[num,bus_name[ndx]] for num,ndx in bus_map.items()}).T
        busndx.columns = ["id","name"]
        busndx.index.name = "bus"
        busndx.to_csv(model.name+"_bus.csv")
        return done

if __name__ == "__main__":
    model = PSSEraw("wecc240_psse.raw")
    done = model.to_pypower()
    print(f"{model.name} Summary")
    print(f"{'-'*len(model.name)}--------")
    for name,data in model.section.items():
        if len(data) > 0:
            print(f"  {name.title()}{'.'*(30-len(name))} {len(data):4d} item{'s' if len(data)>1 else ' '} ({done[name] if name in done else 'ignored'})")

    module = importlib.import_module(model.name)
    case = getattr(module,model.name)()

    if os.path.exists(model.name+"_mods.py"):
        module = importlib.import_module(model.name+"_mods")
        print(f"{model.name} mods loaded",end="...",flush=True)
        case = getattr(module,model.name)(case)
        print("ok")

    print(f"\n{model.name} Check runpf")
    print(f"{'-'*len(model.name)}------------",flush=True)
    runpf(case)

    print(f"\n{model.name} Check runopf")
    print(f"{'-'*len(model.name)}-------------",flush=True)
    runopf(case)

