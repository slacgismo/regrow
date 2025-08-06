"""GridLAB-D optimal powerflow/sizing/placement

Example:

The following example loads the 4-bus model and attempts an OPF. However,
there is insufficient generation to avoid curtailment. Then it runs
the optimal sizing/placement problem and updates the model with the result.
Then the OPF runs without curtailment and the simulation is run with the new model.

>>> import gld
>>> test = Model("test.json")
>>> test.optimal_powerflow()["curtailment"]
>>> test.optimal_sizing(gen_cost=np.array([100,500,1000,1000])+1000j,
                    cap_cost={0:1000,1:500},
                    update_model=True)
>>> test.optimal_powerflow(refresh=True)["curtailment"]
>>> test.run("test_out.json")
"""

import sys
import os
import json
import io
import subprocess
import numpy as np
import numpy.linalg as la
import cvxpy as cp
from typing import Union, Any, TypeVar
import warnings
try:
    from pypower.api import runpf, runopf, ppoption, printpf
except ModuleNotFoundError as err:
    def pypower_api(*args,**kwargs):
        raise RuntimeError(f"pypower not available ({err})")
    runpf = runopf = ppoption = printpf = pypower_api

np.set_printoptions(linewidth=np.inf,formatter={float:lambda x:f"{x:8.4f}"})

pypower = {
    "bus": ["bus_i","type","Pd","Qd","Gs","Bs","area","Vm","Va","baseKV","zone","Vmax","Vmin"],
    "branch": ["fbus","tbus","r","x","b","rateA","rateB","rateC","ratio","angle","status","angmin","angmax"],
    "gen": ["bus","Pg","Qg","Qmax","Qmin","Vg","mBase","status","Pmax","Pmin","Pc1","Pc2","Qc1min","Qc1max","Qc2min","Qc2max","ramp_agc","ramp_10","ramp_30","ramp_q","apf"],
    "gencost": ["model","startup","shutdown"],
    }

def as_case(self) -> dict:

    case = {
        "version": "2",
        "baseMVA": self.globals("pypower::baseMVA"),
        }
    for name,fields in self.pypower.items():
        case[name] = [[self.property(x,y,astype=float) for y in fields] for x in self.find(name)]
    costs = [[float(y) for y in self.property(x,"costs").split(",")] for x in self.find("gencost")]
    for n,cost in enumerate(costs):
        case["gencost"][n].extend([len(cost)]+cost)
    if len(case["gencost"]) == 0:
        del case["gencost"]

    for array in self.pypower:
        if array in case:
            case[array] = np.array(case[array])

    return case

def savecase(self,file):
    """Save pypower case data"""
    with open(file,"w") as fh:
        print(f"""from numpy import array
def {os.path.splitext(os.path.basename(self.name))[0]}():
ppc = {{}}""",file=fh)
        for key,value in self.as_case().items():
            if hasattr(value,"tolist"):
                print(f"""    ppc["{key}"] = array([""",file=fh)
                print(f"""        # {" ".join([f"{x:9.9s}" for x in self.pypower[key]])}""",file=fh)
                for row in value.tolist():
                    print(f"""        [{", ".join([f"{x:8.4g}" for x in row])}],""",file=fh)
                print(f"""    ])""",file=fh)
            else:
                print(f"""    ppc["{key}"] = {value}""",file=fh)

def runpf(self,casedata=None,**kwargs) -> dict:
    """Run pypower powerflow solver"""
    return runpf(self.as_case() if casedata is None else casedata,ppoption(**kwargs))

def runopf(self,casedata=None,**kwargs) -> dict:
    """Run pypower optimal powerflow solver"""
    return runopf(self.as_case() if casedata is None else casedata,ppoption(**kwargs))

def mermaid(self,
    orientation:str="vertical",
    label=None,
    overvolt:float=1.05,
    undervolt:float=0.95,
    highflow:float=1.0,
    showbusdata:Union[bool,list]=False
    ) -> str:
    """Generate network diagram in Mermaid

    Arguments:
    * orientation: horizontal or vertical graph orientation
    * label: property to use as label
    * overvolt: voltage limit for red fill
    * undervolt: voltage limit for blue fill
    * highflow: current limit for heavy line
    * showbusdata: enable display of bus data (or list of properties to display)

    Returns:
    * str: Mermaid diagram string
    """
    orientations = {"vertical":"TB","horizontal":"LR"}
    diagram = [f"""graph {orientations[orientation]}
classDef black fill:#000,stroke:#000;
classDef white fill:#fff,stroke:#000;
classDef red fill:#f64,stroke:#000;
classDef green fill:#0f0,stroke:#000;
classDef blue fill:#6cf,stroke:#000;
"""]
    def _node(bus,spec):
        node = spec["bus_i"]
        name = spec[label] if label else bus
        Vm = self.property(bus,"Vm")
        Pd = self.property(bus,"Pd")
        Qd = self.property(bus,"Qd")
        gens = self.select({"class":"gen","bus":node})
        loads = self.select({"class":"load","parent":bus})
        Pg = sum([self.property(x,"Pg") for x in gens])
        Qg = sum([self.property(x,"Qg") for x in gens])
        shape = "rect" if showbusdata else "fork"
        if isinstance(showbusdata,list):
            busdata = "<br>".join([f"<b><u>{name}</u></b>"]+[f"{x}: {y}" for x,y in spec.items() if x in showbusdata])
        else:
            busdata = "<br>".join([f"<b><u>{name}</u></b>"]+[f"{x}: {y}" for x,y in spec.items() if x in ["id","type","area","Vm","Va","zone"]])
        result = [f"""    {node}@{{shape: {shape}, label: "{busdata}"}}"""]

        if not undervolt is None and Vm < undervolt:
            color = "blue"
        elif not overvolt is None and Vm > overvolt:
            color = "red"
        elif showbusdata:
            color = "white"
        else:
            color = "black"
        result.append(f"""    class {node} {color}""")

        if abs(complex(Pg,Qg)) > 0 or len(loads)>0:
            result.append(f"""    G{node}@{{shape: circle, label: "{name}"}} --{Pg:.1f}{Qg:+.1f}j MVA--> {node}""")
            result.append(f"""    class G{node} white""")
        if abs(complex(Pd,Qd)) > 0:
            result.append(f"""    {node} --{Pd:.1f}{Qd:+.1f}j MVA--> L{node}@{{shape: tri, label: "{name}"}}""")
            result.append(f"""    class L{node} white""")

        return "\n".join(result)

    for bus,spec in self.find("bus").items():

        diagram.append(_node(bus,spec))

    def _line(line,spec):
        current = self.property(line,"current")
        reverse = ( current.real < 0 )
        current = abs(current/1000)
        linetype = "--" if not highflow is None and current < highflow else "=="
        fbus = spec["tbus" if reverse else "fbus"]
        tbus = spec["fbus" if reverse else "tbus"]
        return f"""    {fbus} {linetype}{current:.2f} kA{linetype}> {tbus}"""

    for line,spec in self.find("branch").items():

        diagram.append(_line(line,spec))

    return "\n".join(diagram)
