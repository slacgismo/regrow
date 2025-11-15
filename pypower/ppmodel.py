"""PyPower model accessor

This module defined the PyPower model accessor. Use the bus, branch, gen, gencost, dcline, and dclinecost methods to set the data arrays in the case.
Use the 'case' member to access the case data.

Example:

The following example constructs a new PyPower model and prints the case data.

    model = PPModel()
    print(model.case)
"""

import numpy as np
import pandas as pd

from pypower import idx_bus
from pypower import idx_brch
from pypower import idx_gen
from pypower import idx_cost
from pypower import idx_dcline

from typing import TypeVar

def get_header(idx:TypeVar('module'),*,ignore:list[str]=[]) -> list[str]:
    """Convert idx data to a header list

    Arguments:

    idx: module containing index values

    ignore: list of index values to ignore

    Returns:

    list[str]: ordered list of data array column header names
    """
    mapping = {getattr(idx,x):x for x in dir(idx) if not x.startswith("_") and x not in ignore}
    indexes = sorted(mapping)
    assert max(indexes) - min(indexes) + 1 == len(indexes), "indexes are not strictly sequential"
    return [mapping[n] for n in indexes]

class PPModel:
    """PyPower Model Access"""

    VERBOSE=False
    DEBUG=False

    def __init__(self,
        name:str,
        version:int=2,
        mvabase:float=100.0,
        ):
        """Create PyPower case data

        Arguments:

        name: name of the case

        version: case version number

        mvabase: MVA base value
        """

        if self.DEBUG:
            print(f"DEBUG [PPModel]: creating model {name=} {version=} {mvabase=}")

        self.name = name
        self.case = {
            "version": version,
            "baseMVA": mvabase,
            "bus": [],
            "branch": [],
            "gen" : [],
            "gencost": [],
            "dcline": [],
            "dclinecost": [],
        }

    def with_case(self,case):
        self.case = case
        return self

    def print(self,items=None,file=None):
        """Pretty print case data"""
        if items is None:
            items = ["bus","branch","gen","gencost","dcline","dclinecost"]

        if "bus" in items:
            bus_cols = get_header(idx_bus,ignore=["PQ","PV","REF","NONE"])
            bus = pd.DataFrame(data=self.case["bus"],
                columns=bus_cols[:self.case["bus"].shape[1]])
            bus.index.name="BUS"
            print(bus,file=file)

        if "branch" in items:
            branch_cols = get_header(idx_brch)
            branch = pd.DataFrame(data=self.case["branch"],
                columns=branch_cols[:self.case["branch"].shape[1]])
            branch.index.name="BRANCH"
            print(branch,file=file)

        if "gen" in items:
            gen_cols = get_header(idx_gen)
            gen = pd.DataFrame(data=self.case["gen"],
                columns=gen_cols[:self.case["gen"].shape[1]])
            gen.index.name="GEN"
            print(gen,file=file)

        if "gencost" in items:
            cost_cols = get_header(idx_cost,ignore=["PW_LINEAR","POLYNOMIAL","COST"])
            ncost = self.case["gencost"][idx_cost.NCOST].max()
            cost_cols.extend([f"COST{n}" for n in range(int(ncost))])
            gencost = pd.DataFrame(data=self.case["gencost"],columns=cost_cols)
            gencost.index.name="GENCOST"
            print(gencost,file=file)

        if "dcline" in items and "dcline" in self.case and self.case["dcline"]:
            dcline_cols = get_header(idx_brch)
            dcline = pd.DataFrame(data=self.case["dcline"],
                columns=dcline_cols[:self.case["dcline"].shape[1]])
            dcline.index.name="DCLINE"
            print(dcline,file=file)

        if "dclinecost" in items and "dclinecost" in self.case and self.case["dclinecpst"]:
            cost_cols = get_header(idx_cost,ignore=["PW_LINEAR","POLYNOMIAL","COST"])
            ncost = self.case["dclinecost"][idx_cost.NCOST].max()
            cost_cols.extend([f"COST{n}" for n in range(int(ncost))])
            dclinecost = pd.DataFrame(data=self.case["dclinecost"],columns=cost_cols)
            dclinecost.index.name="DCLINECOST"
            print(dclinecost,file=file)


    def bus(self,**kwargs):
        """Create bus data

        Arguments:

        kwargs: merged bus, load, and shunt data (see pypower.idx_bus for
                details)

        Returns:

        np.array: bus data
        """

        if self.DEBUG:
            print(f"DEBUG [PPModel]: create bus data={kwargs}")

        result = []
        for item in get_header(idx_bus,ignore=["PQ","PV","REF","NONE"]):
            if item in kwargs:
                result.append(kwargs[item])
            elif item not in ["LAM_P","LAM_Q","MU_VMIN","MU_VMAX"]:
                raise ValueError(f"missing {item} data")

        return np.array(result)

    def branch(self,**kwargs):
        """Create branch data

        Arguments:

        kwargs: merged branch and transformer data (see pypower.idx_brch for
                details)

        Returns:

        np.array: bus data
        """
        result = []
        for item in get_header(idx_brch):
            if item in kwargs:
                result.append(kwargs[item])
            elif item not in ["PF","PT","QF","QT","MU_SF","MU_ST","MU_ANGMIN","MU_ANGMAX"]:
                raise ValueError(f"missing {item} data")

        return np.array(result)

    def gen(self,**kwargs):
        """Create gen data

        Arguments:

        kwargs: generation data (see pypower.idx_gen for details)

        Returns:

        np.array: bus data
        """

        if self.DEBUG:
            print(f"DEBUG [PPModel]: create gen data={kwargs}")

        result = []
        for item in get_header(idx_gen):
            if item in kwargs:
                result.append(kwargs[item])
            elif item not in ["MU_PMAX","MU_PMIN","MU_QMAX","MU_QMIN"]:
                raise ValueError(f"missing {item} data")

        return np.array(result)

    def gencost(self,**kwargs):
        """Create gencost data

        Arguments:

        kwargs: generation data (see pypower.idx_gen for details)

        Returns:

        np.array: bus data
        """
        if self.DEBUG:
            print(f"DEBUG [PPModel]: create gencost data={kwargs}")

        result = []
        for item in get_header(idx_cost,ignore=["PW_LINEAR","POLYNOMIAL"]):
            if item in kwargs:
                if kwargs[item].ndim == 1:
                    result.append(kwargs[item])
                else:
                    for col in range(kwargs[item].shape[1]):
                        result.append(kwargs[item][:,col])
            else:
                raise ValueError(f"missing {item} data")

        return np.array(result)

    def dcline(self,**kwargs):
        """Create dcline data
    
        Arguments:


        """
        result = []
        for item in get_header(idx_dcline):
            if item in kwargs:
                result.append(kwargs[item])
            elif item not in ["MU_PMIN","MU_PMAX","MU_QMINF","MU_QMAXF","MU_QMINT","MU_QMAXT"]:
                raise ValueError(f"missing {item} data")

        return np.array(result)

    def dclinecost(self,**kwargs):
        """Create dclinecost data

        Arguments:

        kwargs: generation data (see pypower.idx_gen for details)

        Returns:

        np.array: bus data
        """
        if self.DEBUG:
            print(f"DEBUG [PPModel]: create dclinecost data={kwargs}")

        result = []
        for item in get_header(idx_cost,ignore=["PW_LINEAR","POLYNOMIAL"]):
            if item in kwargs:
                if kwargs[item].ndim == 1:
                    result.append(kwargs[item])
                else:
                    for col in range(kwargs[item].shape[1]):
                        result.append(kwargs[item][:,col])
            else:
                raise ValueError(f"missing {item} data")

        return np.array(result)
