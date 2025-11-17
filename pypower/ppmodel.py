"""PyPower model accessor

This module defined the PyPower model accessor. Use the bus, branch, gen,
gencost, dcline, and dclinecost methods to set the data arrays in the case.
Use the 'case' member to access the case data.

Example:

The following example constructs a new PyPower model and prints the case data.

    model = PPModel()
    print(model.case)
"""

import sys
import datetime as dt
import io
from typing import Self, TypeVar

import numpy as np
import pandas as pd

from pypower import idx_bus
from pypower import idx_brch
from pypower import idx_gen
from pypower import idx_cost

from kml import KML

class idx_dcline:
    """Provide missing column index values that should be in pypower.idx_dcline"""

    # pylint: disable=invalid-name,too-few-public-methods

    F_BUS = 0
    T_BUS = 1
    BR_STATUS = 2
    PF = 3
    PT = 4
    QF = 5
    QT = 6
    VF = 7
    VT = 8
    PMIN = 9
    PMAX = 10
    QMINF = 11
    QMAXF = 12
    QMINT = 13
    QMAXT = 14
    LOSS0 = 15
    LOSS1 = 16
    MU_PMIN = 17
    MU_PMAX = 18
    MU_QMINF = 19
    MU_QMAXF = 20
    MU_QMINT = 21
    MU_QMAXT = 22

def get_header(idx:TypeVar('module'),*,ignore:list[str]=None) -> list[str]:
    """Convert idx data to a header list

    Arguments:

    idx: module containing index values

    ignore: list of index values to ignore

    Returns:

    list[str]: ordered list of data array column header names
    """
    if ignore is None:
        ignore = []
    mapping = {getattr(idx,x):x for x in dir(idx) if not x.startswith("_") and x not in ignore}
    indexes = sorted(mapping)
    assert max(indexes) - min(indexes) + 1 == len(indexes), "indexes are not strictly sequential"
    return [mapping[n] for n in indexes]

class PPModel:
    """PyPower Model Access"""

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

    def set_case(self,
        case:dict,
        ) -> Self:
        """Set the case data

        Arguments:

        case: case data to use

        Returns:

        self: the model with the newly set case data
        """
        self.case = case
        return self

    def save_case(self,
        file:io.StringIO=sys.stdout,
        precision=9,
        ):
        """Save the case data to a file

        Arguments:

        file: file handle to which case data is saved

        precision: float rounding precision
        """
        print(f"""# pypower case '{self.name}' saved on {dt.datetime.now()}
from numpy import array
def {self.name}():
    return {{""",file=file)
        header_map = {
            "bus": idx_bus,
            "branch": idx_brch,
            "gen": idx_gen,
            "gencost": idx_cost,
            "dcline": idx_dcline,
            "dclinecost": idx_cost,
        }
        valid_keys = ["version","baseMVA"] + list(header_map.keys())
        for key,value in [(x,y) for x,y in self.case.items() if x in valid_keys]:
            if isinstance(value,np.ndarray):
                print(f"""      '{key}': array([""",file=file)
                header = ",".join([f"{{0:>{precision+3}s}}".format(x)
                    for x in get_header(header_map[key])])
                print(f"         #{header}",file=file)
                for row in value.tolist():
                    print(f"         [{','.join([f'{{0:{precision+3}g}}'\
                        .format(round(x,precision)) for x in row])}],",file=file)
                print("        ]),",file=file)
            else:
                print(f"""      '{key}': {value},""",file=file)
        print("}",file=file)

    def print(self,
        items=None,
        file=sys.stdout,
        ):
        """Print case data"""
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

        if "dcline" in items and "dcline" in self.case and len(self.case["dcline"]) > 0:
            dcline_cols = get_header(idx_brch)
            dcline = pd.DataFrame(data=self.case["dcline"],
                columns=dcline_cols[:self.case["dcline"].shape[1]])
            dcline.index.name="DCLINE"
            print(dcline,file=file)

        if "gencost" in items:
            cost_cols = get_header(idx_cost,ignore=["PW_LINEAR","POLYNOMIAL","COST"])
            ncost = self.case["gencost"].shape[1] - len(cost_cols)
            print(ncost)
            cost_cols.extend([f"COST{n}" for n in range(int(ncost))])
            gencost = pd.DataFrame(data=self.case["gencost"],columns=cost_cols)
            gencost.index.name="GENCOST"
            print(gencost,file=file)

        if "dclinecost" in items and "dclinecost" in self.case and len(self.case["dclinecost"]) > 0:
            cost_cols = get_header(idx_cost,ignore=["PW_LINEAR","POLYNOMIAL","COST"])
            ncost = self.case["dclinecost"].shape[1] - len(cost_cols)
            cost_cols.extend([f"COST{n}" for n in range(int(ncost))])
            dclinecost = pd.DataFrame(data=self.case["dclinecost"],columns=cost_cols)
            dclinecost.index.name="DCLINECOST"
            print(dclinecost,file=file)

    def to_kml(self,filename):

        kml = KML(filename)
        print(self.case["gis"])

    bus_optional = ["LAM_P","LAM_Q","MU_VMIN","MU_VMAX"]
    @classmethod
    def bus(cls,**kwargs):
        """Create bus data

        Arguments:

        kwargs: merged bus, load, and shunt data (see pypower.idx_bus for
                details)

        Returns:

        np.array: bus data
        """

        header = get_header(idx_bus,ignore=["PQ","PV","REF","NONE"])
        for key,value in kwargs.items():
            if key not in header:
                raise KeyError(f"{key}={value} is not a valid bus item")

        result = []
        for item in header:
            if item in kwargs:
                result.append(kwargs[item])
            elif item not in cls.bus_optional:
                raise ValueError(f"missing {item} data")

        return np.array(result)

    branch_optional = ["PF","PT","QF","QT","MU_SF","MU_ST","MU_ANGMIN","MU_ANGMAX"]
    @classmethod
    def branch(cls,**kwargs):
        """Create branch data

        Arguments:

        kwargs: merged branch and transformer data (see pypower.idx_brch for
                details)

        Returns:

        np.array: bus data
        """
        header = get_header(idx_brch)
        for key,value in kwargs.items():
            if key not in header:
                raise KeyError(f"{key}={value} is not a valid branch item")

        result = []
        for item in header:
            if item in kwargs:
                result.append(kwargs[item])
            elif item not in cls.branch_optional:
                raise ValueError(f"missing {item} data")

        return np.array(result)

    gen_optional = ["MU_PMAX","MU_PMIN","MU_QMAX","MU_QMIN"]
    @classmethod
    def gen(cls,**kwargs):
        """Create gen data

        Arguments:

        kwargs: generation data (see pypower.idx_gen for details)

        Returns:

        np.array: gen data
        """

        result = []
        for item in get_header(idx_gen):
            if item in kwargs:
                result.append(kwargs[item])
            elif item not in cls.gen_optional:
                raise ValueError(f"missing {item} data")

        return np.array(result)

    @classmethod
    def gencost(cls,**kwargs):
        """Create gencost data

        Arguments:

        kwargs: generation data (see pypower.idx_gen for details)

        Returns:

        np.array: cost data
        """

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

    dcline_optional = ["MU_PMIN","MU_PMAX","MU_QMINF","MU_QMAXF","MU_QMINT","MU_QMAXT"]
    @classmethod
    def dcline(cls,**kwargs):
        """Create dcline data

        Arguments:

        kwargs: dcline data (see pypower.idx_dcline for details)

        Returns:

        np.array: dcline data
        """
        result = []
        for item in get_header(idx_dcline):
            if item in kwargs:
                result.append(kwargs[item])
            elif item not in cls.dcline_optional:
                raise ValueError(f"missing {item} data")

        return np.array(result)

    @classmethod
    def dclinecost(cls,**kwargs):
        """Create dclinecost data

        Arguments:

        kwargs: dclinecost data (see pypower.idx_cost for details)

        Returns:

        np.array: cost data
        """

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
