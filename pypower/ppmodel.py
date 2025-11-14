"""PyPower model accessor"""

import numpy as np

from pypower import idx_bus
from pypower import idx_brch
from pypower import idx_gen

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

    def bus(self,**kwargs):
        """Create bus data

        Arguments:

        kwargs: merged bus, load, and shunt data (see pypower.idx_bus for details)

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
        """Create branch data"""
        raise NotImplementedError("PPModel.branch() is not done")

    def gen(self,**kwargs):
        """Create gen data"""

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
        """Create gencost data"""
        raise NotImplementedError("PPModel.gencost() is not done")

    def dcline(self,**kwargs):
        """Create dcline data"""
        raise NotImplementedError("PPModel.dcline() is not done")

    def dclinecost(self,**kwargs):
        """Create dclinecost data"""
        raise NotImplementedError("PPModel.dclinecost() is not done")
