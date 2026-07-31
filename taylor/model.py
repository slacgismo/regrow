import os
import sys
import numpy as np
import importlib.util 
import pypower.runopf as pr
import pypower.ppoption as po
from collections import namedtuple
import cvxpy as cp

class Model:

    basecolumns = {
        "bus": "bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin lambda_P lambda_Q mu_Vmax mu_Vmin",
        "gen": "bus Pg Qg Qmax Qmin Vg mBase status Pmax Pmin Pc1 Pc2 Qc1min Qc1max Qc2min Qc2max ramp_agc ramp_10 ramp_30 ramp_q apf mu_Pmax mu_Pmin mu_Qmax mu_Qmin",
        "branch": "fbus tbus r x b rateA rateB rateC ratio angle status angmin angmax Pf Qf Pt Qt mu_Sf mu_St mu_angmin mu_angmax",    
        "dcline": "fbus tbus status Pf Pt Qf Qt Vf Vt Pmin Pmax Qminf Qmaxf Qmint Qmaxt loss0 loss1",
    }
    column_types = {
        "bus": {
            "bus_i": int,
            "type": int,
            "area": int,
            "zone": int,
        },
        "branch": {
            "fbus": int,
            "tbus": int,
            "status": int,
        },
        "gen": {
            "bus": int,
            "status": int,
        },
        "gencost": {
            "model": int,
            "n": int,
        },
        "dcline": {
            "fbus": int,
            "tbus": int,
            "status": int,
        },
        "dclinecost": {
            "model": int,
            "n": int,
        },
    }
    column_formats = {
        "bus" : {
            "bus_i" : "{:.0f}",
            "type" : "{:.0f}",
            "Pd": "{:.1f}",
            "Qd": "{:.1f}",
            "Gs": "{:.1f}",
            "Bs": "{:.1f}",
            "area": "{:.0f}",
            "Vm": "{:.3f}",
            "Va": "{:.3f}",
            "baseKV": "{:.1f}",
            "zone": "{:.0f}",
            "Vmax": "{:.2f}",
            "Vmin": "{:.2f}",
            "lambda_P": "{:.3f}",
            "lambda_Q": "{:.3f}",
            "mu_Vmax": "{:.3f}",
            "mu_Vmin": "{:.3f}",
        },
        "gen": {
            "bus": "{:.1f}",
            "Pg": "{:.1f}",
            "Qg": "{:.1f}",
            "Qmax": "{:.1f}",
            "Qmin": "{:.1f}",
            "Vg": "{:.2f}",
            "mBase": "{:.1f}",
            "status": "{:.0f}",
            "Pmax": "{:.1f}",
            "Pmin": "{:.1f}",
            "Pc1": "{:.1f}",
            "Pc2": "{:.1f}",
            "Qc1min": "{:.1f}",
            "Qc1max": "{:.1f}",
            "Qc2min": "{:.1f}",
            "Qc2max": "{:.1f}",
            "ramp_agc": "{:.1f}",
            "ramp_10": "{:.1f}",
            "ramp_30": "{:.1f}",
            "ramp_q": "{:.1f}",
            "apf": "{:.2f}",
            "mu_Pmax": "{:.3f}",
            "mu_Pmin": "{:.3f}",
            "mu_Qmax": "{:.3f}",
            "mu_Qmin": "{:.3f}",
            },
        "branch": {
            "fbus": "{:.0f}",
            "tbus": "{:.0f}",
            "r": "{:.5f}",
            "x": "{:.5f}",
            "b": "{:.5f}",
            "rateA": "{:.1f}",
            "rateB": "{:.1f}",
            "rateC": "{:.1f}",
            "ratio": "{:.1f}",
            "angle": "{:.3f}",
            "status": "{:.0f}",
            "angmin": "{:.0f}",
            "angmax": "{:.0f}",
            "Pf": "{:.1f}",
            "Qf": "{:.1f}",
            "Pt": "{:.1f}",
            "Qt": "{:.1f}",
            "mu_Sf": "{:.3f}",
            "mu_St": "{:.3f}",
            "mu_angmin": "{:.3f}",
            "mu_angmax": "{:.3f}",
            },
        "dcline": {
            "fbus": "{:.0f}",
            "tbus": "{:.0f}",
            "status": "{:.0f}",
            "Pf": "{:.1f}",
            "Pt": "{:.1f}",
            "Qf": "{:.1f}",
            "Qt": "{:.1f}",
            "Vf": "{:.1f}",
            "Vt": "{:.1f}",
            "Pmin": "{:.1f}",
            "Pmax": "{:.1f}",
            "Qminf": "{:.1f}",
            "Qmaxf": "{:.1f}",
            "Qmint": "{:.1f}",
            "Qmaxt": "{:.1f}",
            "loss0": "{:.3f}",
            "loss1": "{:.3f}",
            },
    }

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
        self.data = {}

    def graph(self,line=None,node=None):
        graph = ["flowchart LR"]
        aclines = self["branch"]
        bus = self["bus"]
        for n,f,t,s in list(zip(range(len(aclines.fbus)),aclines.fbus,aclines.tbus,aclines.status)):
            if isinstance(node,list):
                from_label = f'"{node[f-1]}"'
                to_label = f'"{node[t-1]}"'
            elif isinstance(node,str):
                from_label = ('"'+self.column_formats["bus"][node]+'"').format(getattr(bus,node)[f-1]) if node else f
                to_label = ('"'+self.column_formats["bus"][node]+'"').format(getattr(bus,node)[t-1]) if node else t
            else:
                from_label = f
                to_label = t
            if isinstance(line,list):
                line_label = f'"{line[n]}"'
            elif isinstance(line,str):
                line_label = ('"'+self.column_formats["branch"][line]+'"').format(getattr(aclines,line)[n])
            else:
                line_label = n+1
            if s:
                graph.append(f"  {f:.0f}(({from_label})) == {line_label} ==> {t:.0f}(({to_label}))")
        dclines = self["dcline"]
        for n,f,t,s in list(zip(range(len(dclines.fbus)),dclines.fbus,dclines.tbus,dclines.status)):
            from_label = ('"'+self.column_formats["bus"][node]+'"').format(getattr(bus,node)[f-1]) if node else f
            to_label = ('"'+self.column_formats["bus"][node]+'"').format(getattr(bus,node)[t-1]) if node else t
            line_label = ('"'+self.column_formats["branch"][line]+'"').format(getattr(aclines,line)[n]) if line else (n+1)
            if s:
                graph.append(f"  {f:.0f}(({from_label})) -- {line_label} -- {t:.0f}(({to_label}))")
        return "\n".join(graph)

    def cost(self,Pg=None):

        bus = self["bus"]
        gen = self["gen"]
        gencost = self["gencost"]
        result = np.zeros(Pg.shape)
        for i,p in enumerate(gen.Pg if Pg is None else Pg):
            bus_i = bus.bus_i[i]
            model = gencost.model[i]
            n = gencost.n[i]
            if model == 1: # PWLF
                pwlf = self.self.model["gencost"][i,-1:-n-1:-1]
                # print(f"{pwlf=}")
                raise NotImplementedError(f"pwlf {model=} not implemented for {bus_i=}")
            elif model == 2: # polynomial
                # print(f"{self.model['gencost']=}")
                poly = self.model["gencost"][i,-1:-n-1:-1]
                # print(f"{poly=}")
                result[i] = np.polyval(poly,p)
            else:
                raise RuntimeError(f"{model=} is an invalid gencost model on {bus_i=}")
        return result

    def set_result(self,result):
        self.result = result
        _ncost = 2 if min(self.result["gencost"].T[0]) == 1 else 1
        self.columns = dict(self.basecolumns)
        self.columns.update({
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
        })
        self.data = {}

    def solve_pf(self,**kwargs):
        if "VERBOSE" not in kwargs:
            kwargs["VERBOSE"] = 0
        if "OUT_ALL" not in kwargs:
            kwargs["OUT_ALL"] = 0
        self.options = po.ppoption(**kwargs)
        self.set_result(pr.runpf(self.model,self.options))
        return self.result["success"]

    def solve_opf(self,**kwargs):
        if "linearize" in kwargs:
            if kwargs["linearize"] == "powerflow":

                bus = self["bus"]
                branch = self["branch"]
                gen = self["gen"]
                gencost = self["gencost"]
                N = len(bus.bus_i)
                M = len(branch.fbus)
                G = len(gen.bus)

                P, Smax = np.zeros((N, N)), np.zeros((N,N))
                B = np.zeros((N, N))
                fbus = [n-1 for n in branch.fbus]
                tbus = [n-1 for n in branch.tbus]
                pmax = (Model.coarray((N,M),fbus,branch.rateA)+Model.coarray((N,M),tbus,branch.rateA)).sum(axis=1)
                pmin = bus.Pd - pmax
                pmax = Model.coarray(N,gen.bus,gen.Pmax) + pmax
                for i, j, b, pt, pf, smax in list(
                    zip(
                        *[
                            getattr(branch, x)
                            for x in ["fbus", "tbus","b", "Pf", "Pt", "rateA"]
                        ]
                    )
                ):
                    i, j = i - 1, j - 1
                    B[i, j] =  B[j, i] = b
                    P[i,j], P[j,i] = pt, pf
                    Smax[i,j] = Smax[j,i] = smax
                B = cp.Parameter(shape=(N,N),value=B)
                P = cp.Parameter(shape=(N,N),value=P)
                p = P.sum(axis=1)

                v = cp.Variable(N)
                g = cp.Variable(N)
                C = gencost.c0*g**2 + gencost.c1*g + gencost.c2

                objective = cp.Minimize(C)
                
                self.result = {
                    "success":False,
                    "raw" : {
                        "output" : {
                            "message" : f"linearize='{kwargs['linearize']}' is not implemented yet"
                        }
                    }
                }

            elif kwargs["linearize"] == "decoupled":
                
                self.result = {
                    "success":False,
                    "raw" : {
                        "output" : {
                            "message" : f"linearize='{kwargs['linearize']}' is not implemented yet"
                        }
                    }
                }

            elif kwargs["linearize"] == "transport":
                
                self.result = {
                    "success":False,
                    "raw" : {
                        "output" : {
                            "message" : f"linearize='{kwargs['linearize']}' is not implemented yet"
                        }
                    }
                }

            else:

                raise ValueError(f"linearize='{kwargs['linearize']}' is not valid")

        elif "relaxation" in kwargs:

                self.result = {
                    "success":False,
                    "raw" : {
                        "output" : {
                            "message" : f"relaxations is not implemented yet"
                        }
                    }
                }

        else:

            if "VERBOSE" not in kwargs:
                kwargs["VERBOSE"] = 0
            if "OUT_ALL" not in kwargs:
                kwargs["OUT_ALL"] = 0
            self.options = po.ppoption(**kwargs)
            self.set_result(pr.runopf(self.model,self.options))

        return self.result["success"]

    def __getitem__(self,field):
        if field in self.data:
            return self.data[field]
        if self.result is None:
            return None
        result = namedtuple(field, self.columns[field].split())(
            *self.result[field].T 
            if field in self.result 
            else [[]] * len(self.columns[field].split())
        )
        if field in self.column_types:
            for name,dtype in self.column_types[field].items():
                if len(getattr(result,name)) > 0:
                    result = result._replace(**{name:getattr(result,name).astype(dtype)})
        self.data[field] = result
        return result

    @staticmethod
    def coarray(shape,n,x,dtype=None):
        result = np.zeros(shape=shape,dtype=dtype if dtype else x.dtype if hasattr(x,'dtype') else float)
        for _n,_x in zip(n,x):
            result[_n] = _x
        return result

if __name__ == "__main__":

    test = Model(os.path.join(os.getcwd(),"case9.py"))

    test.solve_opf()
    print("baseline OPF",test.result,sep="...\n")

    test.solve_opf(linearize='powerflow')
    print("Linearized powerflow OPF",test.result,sep="...\n")

    test.solve_opf(linearize='decoupled')
    print("decoupled powerflow OPF",test.result,sep="...\n")

    test.solve_opf(linearize='transport')
    print("transport powerflow OPF",test.result,sep="...\n")


