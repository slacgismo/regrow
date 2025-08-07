import os
import sys
import importlib.util 
import pypower.runopf as pr
import pypower.ppoption as po
from collections import namedtuple

class Model:

    def __init__(self,pathname):
        self.path,self.file = os.path.split(pathname)
        self.name = os.path.splitext(self.file)[0]
        self.spec = importlib.util.spec_from_file_location(self.name,pathname)
        self.module = importlib.util.module_from_spec(self.spec)
        sys.modules[self.name] = self.module
        self.spec.loader.exec_module(self.module)
        self.model = getattr(self.module,self.name)()
        self.options = po.ppoption()
        self.result = None

    def graph(self):
        graph = ["flowchart LR"]
        aclines = self["branch"]
        for n,f,t,s in list(zip(range(len(aclines.fbus)),aclines.fbus,aclines.tbus,aclines.status)):
            if s:
                graph.append(f"  {f:.0f}(({f:.0f})) == {n+1:.0f} ==> {t:.0f}(({t:.0f}))")
        dclines = self["dcline"]
        for n,f,t,s in list(zip(range(len(dclines.fbus)),dclines.fbus,dclines.tbus,dclines.status)):
            if s:
                graph.append(f"  {f:.0f}(({f:.0f})) -- {n+1:.0f} --> {t:.0f}(({t:.0f}))")
        return "\n".join(graph)

    def solve_pf(self,**kwargs):
        if "VERBOSE" not in kwargs:
            kwargs["VERBOSE"] = 0
        if "OUT_ALL" not in kwargs:
            kwargs["OUT_ALL"] = 0
        self.options = po.ppoption(**kwargs)
        self.result = pr.runpf(self.model,self.options)
        return self.result["success"]

    def solve_opf(self,**kwargs):
        if "VERBOSE" not in kwargs:
            kwargs["VERBOSE"] = 0
        if "OUT_ALL" not in kwargs:
            kwargs["OUT_ALL"] = 0
        self.options = po.ppoption(**kwargs)
        self.result = pr.runopf(self.model,self.options)
        return self.result["success"]

    def __getitem__(self,x):
        if self.result is None:
            return None
        _ncost = 2 if min(self.result["gencost"].T[0]) == 1 else 1
        _columns = {
            "bus": "bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin lambda_P lambda_Q mu_Vmax mu_Vmin",
            "gen": "bus Pg Qg Qmax Qmin Vg mBase status Pmax Pmin Pc1 Pc2 Qc1min Qc1max Qc2min Qc2max ramp_agc ramp_10 ramp_30 ramp_q apf mu_Pmax mu_Pmin mu_Qmax mu_Qmin",
            "branch": "fbus tbus r x b rateA rateB rateC ratio angle status angmin angmax Pf Qf Pt Qt mu_Sf mu_St mu_angmin mu_angmax",
            "gencost": "model startup shutdown n "
            + " ".join(
                [
                    f"c{x}"
                    for x in range(
                        int(
                            max(self.result["gencost"].T[3]) if "gencost" in self.result else 0
                        )
                        * _ncost
                        - 1,
                        -1,
                        -1,
                    )
                ]
            ),
            "dcline": "fbus tbus status Pf Pt Qf Qt Vf Vt Pmin Pmax Qminf Qmaxf Qmint Qmaxt loss0 loss1",
            "dclinecost": "model startup shutdown n "
            + " ".join(
                [
                    f"c{x}"
                    for x in range(
                        int(
                            max(self.result["dclinecost"].T[3])
                            if "dclinecost" in self.result
                            else 0
                        )
                        * 5
                        - 1,
                        -1,
                        -1,
                    )
                ]
            ),
        }
        return namedtuple(x, _columns[x].split())(
            *self.result[x].T if x in self.result else [[]] * len(_columns[x].split())
        )

if __name__ == "__main__":

    test = Model(os.path.join(os.getcwd(),"case14.py"))
