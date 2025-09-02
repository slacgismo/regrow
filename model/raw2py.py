"""Convert PSSE "raw" file to PyPOWER "py" file"""

# Copyright (C) 2025, Eudoxys Sciences LLC
# Author: dchassin@eudoxys.com
# Project: REGROW (NREL-SUB-2025-10301)

DEBUG = True # raise pypower exceptions instead of just printing error message
NOCHECK = False # skip value checks on pypower case data

import os
import sys
import datetime as dt
import json
import importlib
import numpy as np
import pandas as pd

import pypower.idx_bus as pp_bus
import pypower.idx_brch as pp_branch
import pypower.idx_gen as pp_gen
import pypower.idx_gen as pp_gencost
import pypower.idx_dcline as pp_dcline
import pypower.idx_dcline as pp_dclinecost
from pypower.runpf import runpf
from pypower.runopf import runopf

import psse

pp_index = {
        "bus" : ['BUS_I','BUS_TYPE','PD','QD','GS','BS','BUS_AREA','VM','VA','BASE_KV','ZONE','VMAX','VMIN','LAM_P','LAM_Q','MU_VMAX','MU_VMIN'],
        "branch": ['F_BUS','T_BUS','BR_R','BR_X','BR_B','RATE_A','RATE_B','RATE_C','TAP','SHIFT','BR_STATUS','ANGMIN','ANGMAX','PF','QF','PT','QT','MU_SF','MU_ST','MU_ANGMIN','MU_ANGMAX'],
        "gen": ['GEN_BUS','PG','QG','QMAX','QMIN','VG','MBASE','GEN_STATUS','PMAX','PMIN','PC1','PC2','QC1MIN','QC1MAX','QC2MIN','QC2MAX','RAMP_AGC','RAMP_10','RAMP_30','RAMP_Q','APF','MU_PMAX','MU_PMIN','MU_QMAX','MU_QMIN'],
        "gencost": ['MODEL','STARTUP','SHUTDOWN','NCOST','COST'],
        "dcline": ['F_BUS','T_BUS','BR_STATUS','PF','PT','QF','QT','VF','VT','PMIN','PMAX','QMINF','QMAXF','QMINT','QMAXT','LOSS0','LOSS1'],
        "dclinecost": ['MODEL','STARTUP','SHUTDOWN','NCOST','COST'],
    }
 
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
                if len(data[section]) == 0 or line[1] == ' ': #len(data[section]) == 0 or len(data[section][-1])+len(values) >= len(self.headings[section]):
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
            # "dcline": [],
            # "dclinecost": [],
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
        N = len(bus_index)
        done["BUS DATA"] = "ok"

        # loads
        Pd = {}
        Qd = {}
        for n,m in enumerate(self.section["LOAD DATA"]):
            bus_i = m[psse.bus.I]

            if bus_i not in Pd:
                Pd[bus_i] = np.array([0.0,0.0,0.0])
            Pd[bus_i] += [m[psse.load.PL],m[psse.load.IP],m[psse.load.YP]] if m[psse.load.STAT] else [0.0,0.0,0.0]

            if bus_i not in Qd:
                Qd[bus_i] = np.array([0.0,0.0,0.0])
            Qd[bus_i] += [m[psse.load.QL],m[psse.load.IQ],m[psse.load.YQ]] if m[psse.load.STAT] else [0.0,0.0,0.0]

            if sum(Pd[bus_i]) < 0:
                BUS_I = bus_map[bus_i]+1
                print(f"WARNING: load {n=} ({bus_name[BUS_I]}#{bus_i}) power is negative Pd={dict(zip('ZIP',Pd[bus_i]))}",flush=True,file=sys.stderr)

        # print("LOAD:",round(sum([abs(complex(sum(x),sum(y))) for x,y in zip(Pd.values(),Qd.values())]),1),"MVA")
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
            fbus = bus_map[m[psse.branch.I]]
            tbus = bus_map[abs(m[psse.branch.J])] # negative J means metered branch

            FBUS = round(fbus + 1)
            TBUS = round(tbus + 1)
            BR_R = round(m[psse.branch.R],9)
            BR_X = round(m[psse.branch.X],9)
            BR_B = round(m[psse.branch.B],9)
            RATE_A,RATE_B,RATE_C = sorted(m[psse.branch.RATE1:psse.branch.RATE3+1])
            TAP = 0
            SHIFT = 0.0
            BR_STATUS = round(m[psse.branch.STAT])
            ANGMIN = -360
            ANGMAX = +360

            if not NOCHECK:
                assert 0 < FBUS <= N, f"branch {n=} refers to invalid bus {FBUS=}"
                assert 0 < TBUS <= N, f"branch {n=} refers to invalid bus {TBUS=}"
                assert BR_R > 0, f"branch {n=} value {BR_R=} is not positive"
                # negative BR_X is ok
                assert BR_B >= 0, f"branch {n=} value {BR_B=} is not positive"
                assert 0 <= RATE_A <= RATE_B <= RATE_C, f"branch {n=} rates {RATE_A=}, {RATE_B=}, {RATE_C=} are not ordered correctly"
                assert 0 <= TAP <= 2, f"branch {n=} value {TAP=} is outside normal range 0.0 to 2.0"
                assert -180 <= SHIFT <= +180, f"branch {n=} value {SHIFT=} is outside range -180 to +180"
                assert BR_STATUS in [0,1], f"branch {n=} value {BR_STATUS=} is not 0 or 1"
                assert -360 <= ANGMIN <= +360, f"branch {n=} value {ANGMIN=} is outside range -360 to +360"
                assert -360 <= ANGMAX <= +360, f"branch {n=} value {ANGMAX=} is outside range -360 to +360"
                assert ANGMIN <= ANGMAX, f"branch {n=} value {ANGMIN=} is greater than {ANGMAX=}"

            case["branch"].append([FBUS,TBUS,BR_R,BR_X,BR_B,RATE_A,RATE_B,RATE_C,TAP,SHIFT,BR_STATUS,ANGMIN,ANGMAX])

            if BR_STATUS == 1: # include branch shunt in from and to busses
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
            else:
                print(f"WARNING: branch {n} (from {bus_name[FBUS]}#{m[psse.branch.I]} to {bus_name[TBUS]}#{m[psse.branch.J]}) not in service ({BR_STATUS=}), shunts not included in busses {fbus} and {tbus}",file=sys.stderr)
        done["BRANCH DATA"] = "ok"

        # transformers
        Zbase = (np.array([x[psse.bus.BASEKV] for x in self.section["BUS DATA"]])**2 / self.mvabase).tolist()
        for n,m in enumerate(self.section["TRANSFORMER DATA"]):

            if m[psse.transformer.K] > 0:
                print(f"WARNING: three-winding transformer {n} is not supported -- third winding ignored")

            fbus = bus_map[m[psse.transformer.I]]
            tbus = bus_map[abs(m[psse.transformer.J])]
            Zbf = Zbase[fbus]

            FBUS = round(fbus + 1)
            TBUS = round(tbus + 1)
            BR_R = round(m[psse.transformer.R12] / Zbf,9)
            BR_X = round(m[psse.transformer.X12] / Zbf,9)
            BR_B = 0.0
            RATE_A = m[psse.transformer.RATE11]
            RATE_B = m[psse.transformer.RATE12]
            RATE_C = m[psse.transformer.RATE13]
            RATIO = 0
            SHIFT = 0.0
            BR_STATUS = round(m[psse.transformer.STAT])
            ANGMIN = -360
            ANGMAX = +360

            if not NOCHECK:
                assert 0 < FBUS <= N, f"branch {n=} refers to invalid bus {FBUS=}"
                assert 0 < TBUS <= N, f"branch {n=} refers to invalid bus {TBUS=}"
                assert BR_R > 0, f"branch {n=} value {BR_R=} is not positive"
                # negative BR_X is ok
                assert BR_B >= 0, f"branch {n=} value {BR_B=} is not positive"
                assert 0 <= RATE_A <= RATE_B <= RATE_C, f"branch {n=} rates {RATE_A=}, {RATE_B=}, {RATE_C=} are not ordered correctly"
                assert 0 <= TAP <= 2, f"branch {n=} value {TAP=} is outside normal range 0.0 to 2.0"
                assert -180 <= SHIFT <= +180, f"branch {n=} value {SHIFT=} is outside range -180 to +180"
                assert BR_STATUS in [0,1], f"branch {n=} value {BR_STATUS=} is not 0 or 1"
                assert -360 <= ANGMIN <= +360, f"branch {n=} value {ANGMIN=} is outside range -360 to +360"
                assert -360 <= ANGMAX <= +360, f"branch {n=} value {ANGMAX=} is outside range -360 to +360"
                assert ANGMIN <= ANGMAX, f"branch {n=} value {ANGMIN=} is greater than {ANGMAX=}"

            status = m[psse.transformer.STAT]
            case["branch"].append([FBUS,TBUS,BR_R,BR_X,BR_B,RATE_A,RATE_B,RATE_C,RATIO,SHIFT,BR_STATUS,ANGMIN,ANGMAX])

        done["TRANSFORMER DATA"] = "ok"

        # busses
        for n,m in enumerate(self.section["BUS DATA"]):

            bus_i = m[psse.bus.I]
            p = Pd[bus_i] if bus_i in Pd else [0.0,0.0,0.0]
            q = Qd[bus_i] if bus_i in Qd else [0.0,0.0,0.0]
            g = Gs[bus_i] if bus_i in Gs else 0.0
            b = Bs[bus_i] if bus_i in Bs else 0.0
            vm = m[psse.bus.VM]

            BUS_I = round(bus_map[bus_i] + 1)
            BUS_TYPE = m[psse.bus.BUSTYPE]
            PD = float(round(p[0]+(p[1]+p[2]*vm)*vm,1))
            QD = float(round(q[0]+(q[1]+q[2]*vm)*vm,1))
            GS = g
            BS = b
            BUS_AREA = round(m[psse.bus.AREA])
            VM = vm
            VA = m[psse.bus.VA]
            BASE_KV = m[psse.bus.BASEKV]
            ZONE = round(m[psse.bus.ZONE])
            VMAX = m[psse.bus.VMAX]
            VMIN = m[psse.bus.VMIN]

            if not NOCHECK:
                assert 0 < BUS_I <= N, f"bus {n=} value {BUS_I=} is invalid"
                assert BUS_TYPE in [pp_bus.PQ,pp_bus.PV,pp_bus.REF,pp_bus.NONE], f"bus {n=} value {BUS_TYPE=} in invalid"
                assert 0 <= GS, f"bus {n=} value {GS=} is negative"
                assert 0 <= BS, f"bus {n=} value {BS=} is negative"
                assert 0 < BUS_AREA, f"bus {n=} value {BUS_AREA=} is not positive"
                assert 0.5 <= VM <= 1.5, f"bus {n=} value {VM=} is not between 0.5 and 1.5"
                assert -90 < VA < 90, f"bus {n=} value {VA=} is not between -90 and 90"

            case["bus"].append([BUS_I,BUS_TYPE,PD,QD,GS,BS,BUS_AREA,VM,VA,BASE_KV,ZONE,VMAX,VMIN])
        
        done["BUS DATA"] = "ok"

        # generators
        if not os.path.exists(model.name+"_gentypes.csv"):
            with open(model.name+"_gentypes.csv","w") as fh:
                print("type,c2,c1,c0",file=fh)
                for gt in sorted({x[psse.gen.ID] for x in self.section["GENERATOR DATA"]}):
                    print(f"{gt},0,0,0",file=fh)
        gentypes = pd.read_csv(model.name+"_gentypes.csv",index_col=0).to_dict('index')
        for n,m in enumerate(self.section["GENERATOR DATA"]):

            bus_i = bus_map[m[psse.gen.I]]

            GEN_BUS = round(bus_i + 1)
            PG = m[psse.gen.PG]
            QG = m[psse.gen.QG]
            QMAX = m[psse.gen.QT]
            QMIN = m[psse.gen.QB]
            VG = m[psse.gen.VS]
            MBASE = m[psse.gen.MBASE]
            GEN_STATUS = m[psse.gen.STAT]
            PMAX = m[psse.gen.PT]
            PMIN = m[psse.gen.PB]
            PC1 = 0.0
            PC2 = 0.0
            QC1MIN = 0.0
            QC1MAX = 0.0
            QC2MIN = 0.0
            QC2MAX = 0.0
            RAMP_AGC = 0.0
            RAMP_10 = 0.0
            RAMP_30 = 0.0
            RAMP_Q = 0.0
            APF = 0.0

            if not NOCHECK:
                assert 0 < GEN_BUS <= N, f"gen {n=} value {GEN_BUS=} is invalid"
                assert QMIN <= QMAX, f"gen {n=} value {QMIN=} greater than {QMAX=}"
                assert 0.5 <= VG <= 1.5, f"gen {n=} value {VG=} is not between 0.5 and 1.5"
                assert 0 <= MBASE, f"gen {n=} value {MBASE=} is negative"
                assert GEN_STATUS in [0,1], f"gen {n=} is not 0 or 1"
                assert 0 <= PC1, f"gen {n=} value {PC1=} is invalid"
                assert 0 <= PC2, f"gen {n=} value {PC2=} is invalid"
                assert PC1 <= PC2, f"gen {n=} value {PC1=} greater than {PC2=}"
                assert 0 <= QC1MIN, f"gen {n=} value {QC1MIN=} is invalid"
                assert 0 <= QC1MAX, f"gen {n=} value {QC1MAX=} is invalid"
                assert QC1MIN <= QC1MAX, f"gen {n=} value {QC1MIN=} greater than {QC1MAX=}"
                assert 0 <= QC2MIN, f"gen {n=} value {QC2MIN=} is invalid"
                assert 0 <= QC2MAX, f"gen {n=} value {QC2MAX=} is invalid"
                assert QC2MIN <= QC2MAX, f"gen {n=} value {QC2MIN=} greater than {QC2MAX=}"
                assert 0 <= RAMP_AGC, f"gen {n=} value {RAMP_AGC=} is invalid"
                assert 0 <= RAMP_10, f"gen {n=} value {RAMP_10=} is invalid"
                assert 0 <= RAMP_30, f"gen {n=} value {RAMP_30=} is invalid"
                assert 0 <= RAMP_Q, f"gen {n=} value {RAMP_Q=} is invalid"
                assert 0 <= APF, f"gen {n=} value {APF=} is invalid"

            if PG < 0 :
                print(f"WARNING: generator {n=} ({bus_name[GEN_BUS]}#{m[psse.gen.I]}) value {PG=} is negative",flush=True,file=sys.stderr)

            if PMIN < 0:
                print(f"WARNING: generator {n=} ({bus_name[GEN_BUS]}#{m[psse.gen.I]}) value {PMIN=} is negative",file=sys.stderr)

            case["gen"].append([GEN_BUS,PG,QG,QMAX,QMIN,VG,MBASE,GEN_STATUS,PMAX,PMIN,PC1,PC2,QC1MIN,QC1MAX,QC2MIN,QC2MAX,RAMP_AGC,RAMP_10,RAMP_30,RAMP_Q,APF])
            genname = f"{self.section['BUS DATA'][bus_i][psse.bus.NAME]}_{m[psse.gen.ID]}"

            # gencost
            try:
                p = [self.gencost[genname][x] for x in ["model","startup","shutdown","cost"]]
            except KeyError:
                p = [pp_gencost.POLYNOMIAL,0.0,0.0,",".join([str(x) for x in gentypes[m[psse.gen.ID]].values()])]
                self.gencost[genname] = p                
            d = [float(x) for x in p[3].split(",")]
            gencost = [p[0],p[1],p[2],len(d)] + d
            case["gencost"].append(gencost)

        # print("GENS:",sum([abs(complex(x[2],x[3])) for x in case["gen"]]),"MVA")
        done["GENERATOR DATA"] = "ok"

        # dclines
        for n,m in enumerate(self.section["TWO-TERMINAL DC DATA"]):

            raise RuntimeError("DC line imports not supported yet")

            # case["dcline"].append(TODO)
            # case["dclinecost"].append([2,0,0,2,0,0,0])

        done["TWO-TERMINAL DC DATA"] = "TODO"

        # load mods
        if os.path.exists(model.name+"_mods.py"):
            module = importlib.import_module(model.name+"_mods")
            print(f"{model.name}_mods.py",end="...",flush=True,file=sys.stderr)
            case = getattr(module,model.name)(case)
            print("ok",file=sys.stderr)

        # write case file
        with open(file if file else self.name+".py","w") as fh:
            print(f"# Generated by '{' '.join(sys.argv)}' from '{model.name}.raw' at {dt.datetime.now()}",file=fh)
            print(f"from numpy import array, float64",file=fh)
            print(f"def {self.name}():",file=fh)
            print(f"   return {{",file=fh)
            for tag,data in case.items():
                if hasattr(data,"tolist"):
                    data = data.tolist() # change np.array to list
                if isinstance(data,list):
                    print(f"""    "{tag}": array([""",file=fh)
                    if tag in pp_index:
                        print("      #",",".join([f"{x:>10.10s}" for x in pp_index[tag]]),file=fh)
                    for row in data:
                        print(f"""      [ {','.join([f'{x:10.5g}' for x in row])}],""",file=fh)
                    print("    ],dtype=float64),",file=fh)
                else:
                    print(f"""    "{tag}": {repr(data)},""",file=fh)
            print(f"}}",file=fh)

        # force all lists to np.array
        for tag,data in case.items():
            if isinstance(data,list):
                case[tag] = np.array(data)

        # write cost file
        if not os.path.exists(self.name+"_cost.csv") or os.path.getmtime(self.name+"_gentypes.csv") > os.path.getmtime(self.name+"_cost.csv"):
            df = pd.DataFrame(self.gencost).T
            df.columns = ["model","startup","shutdown","cost"]
            df.index.name = "id"
            df.to_csv(self.name+"_cost.csv",header=True,index=True)

        # save bus map
        busndx = pd.DataFrame({ndx:[num,bus_name[ndx]] for num,ndx in bus_map.items()}).T
        busndx.columns = ["id","name"]
        busndx["bus_i"] = busndx.index + 1
        busndx.index.name = "bus"
        busndx.to_csv(model.name+"_bus.csv")

        self.case = case

        return done

    def validate(self):

        print("\n"+self.name,"Validation",file=sys.stderr)
        print("-"*len(self.name) + "-" + "-"*len("validation"),file=sys.stderr)

        # get network data
        nodes = self.case["bus"]
        lines = self.case["branch"]
        N = len(nodes)
        M = len(lines)

        # check branches
        for line in lines:
            assert 0 < line[pp_branch.F_BUS] <= N and 0 < line[pp_branch.T_BUS] <= N, "invalid bus reference"

        # check incidence matrix
        B = np.zeros((N,M))
        for m,line in enumerate(lines):
            B[round(line[pp_branch.F_BUS])-1,m] = 1
            B[round(line[pp_branch.T_BUS])-1,m] = 1
        unconnected_nodes = [n for n,x in enumerate(B.sum(axis=1).astype(int).tolist()) if x == 0]
        if unconnected_nodes:
            print(f"WARNING: {len(unconnected_nodes)} unconnected nodes: {unconnected_nodes}",file=sys.stderr)
        else:
            print("No unconnected node -- ok",file=sys.stderr)

        # check islands
        networks = sum([1 for x in np.linalg.eig(B@B.T)[0].real.round(6) if x==0])-len(unconnected_nodes)
        if networks > 1:
            print("WARNING:",networks-1,"isolated networks found (not counting unconnected nodes)",file=sys.stderr)
        else:
            print("No network islands found -- ok",file=sys.stderr)

        # check generation
        total_gen = 0
        for n,m in enumerate(self.case["gen"]):
            total_gen += complex(m[pp_gen.PG],m[pp_gen.QG])
            assert m[pp_gen.PMIN] <= m[pp_gen.PG] <= m[pp_gen.PMAX], f"generator {n} real power out of range"
            assert m[pp_gen.QMIN] <= m[pp_gen.QG] <= m[pp_gen.QMAX], f"generator {n} reactive power out of range"
        print(f"Total generation: {abs(total_gen):.1f} MVA",file=sys.stderr)

        # check load
        total_load = 0
        for m in self.case["bus"]:
            total_load += complex(m[pp_bus.PD],m[pp_bus.QD])
        print(f"Total load: {abs(total_load):.1f} MVA",file=sys.stderr)

        if abs(total_gen) < abs(total_load):
            print("WARNING: insufficient generation",file=sys.stderr)
        else:
            print("Power balance is ok",file=sys.stderr)

        # check costs
        if "gencost" in self.case:
            assert len(self.case["gen"]) == len(self.case["gencost"]), "gencost does not match gen size"
        if "dcline" in self.case:
            assert len(self.case["dcline"]) == len(self.case["dclinecost"]), "dclinecost does not match dcline size"

        # verify that case file saved correctly
        module = importlib.import_module(model.name)
        case = getattr(module,model.name)()
        for tag,data in case.items():
            if type(data) is np.array:
                assert (case[tag] == self.case[tag]).all(), "case file does not match model"
    
        print("Validation ok",file=sys.stderr,flush=True)


if __name__ == "__main__":

    model = PSSEraw("wecc240_psse.raw")
    sys.argv = [os.path.basename(__file__)]

    done = model.to_pypower()

    # module = importlib.import_module(model.name)
    # case = getattr(module,model.name)()
    # assert case == self.case, "case file does not match model"
    
    model.validate()

    print(f"\n{model.name} Summary")
    print(f"{'-'*len(model.name)}--------")
    for name,data in model.section.items():
        if len(data) > 0:
            print(f"  {name.title()}{'.'*(30-len(name))} {len(data):4d} item{'s' if len(data)>1 else ' '} ({done[name] if name in done else 'ignored'})")

    failed = 0

    # test solvers
    print(f"\n{model.name} Check runopf")
    print(f"{'-'*len(model.name)}-------------",flush=True)
    try:
        result = runopf(model.case)
    except:
        failed += 1
        e_type,e_value,e_trace = sys.exc_info()
        print(f"ERROR [raw2py]: runopf failed, {e_type.__name__} {e_value}")
        if DEBUG:
            raise

    print(f"\n{model.name} Check runpf")
    print(f"{'-'*len(model.name)}------------",flush=True)
    try:
        result,ok = runpf(model.case)
    except:
        e_type,e_value,e_trace = sys.exc_info()
        failed += 1
        print(f"ERROR [raw2py]: runpf failed, {e_type.__name__} {e_value}")
        if DEBUG:
            raise
    if not ok:
        print(f"ERROR [raw2py]: runpf did not converge")

    exit(failed)