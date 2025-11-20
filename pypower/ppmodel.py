"""PyPower model accessor

This module defined the PyPower model accessor. Use the bus, branch, gen,
gencost, dcline, and dclinecost methods to set the data arrays in the case.
Use the 'case' member to access the pypower case data.

The `save_case()` method is used to export a PyPower case file. 

The `save_kml()` method is used to export a Google Earth KML file. 

The `print()` method is used to output the case data in human readable form
using a Pandas data frame.

Example:

The following example constructs a new PyPower model and prints the case data.

    model = PPModel()
    print(model.case)
"""

import sys
import datetime as dt
import io
from typing import Self, TypeVar, Callable

import numpy as np
import pandas as pd

from pypower import idx_bus
from pypower import idx_brch as idx_branch
from pypower import idx_gen
from pypower import idx_cost as idx_gencost
from pypower import idx_cost as idx_dclinecost

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

class idx_gis:
    """Provide column index values for GIS data"""

    # pylint: disable=invalid-name,too-few-public-methods

    BUS_I = 0 # bus index
    LAT = 1 # bus latitude
    LON = 2 # bus longitude
    GEOHASH = 3 # bus node id
    NAME = 4 # bus name
    GEN = 5 # generator count (nan: no gen allowed)
    LOAD = 6 # load count (nan: no load allowed)

ignore_idx = {
    "bus": ["PQ","PV","REF","NONE"],
    "branch": [],
    "gen": [],
    "gencost": ["PW_LINEAR","POLYNOMIAL"],
    "dcline": [],
    "dclinecost": ["PW_LINEAR","POLYNOMIAL"],
    "gis": [],
}


def get_header(name:str,*,ignore:list[str]=None) -> list[str]:
    """Convert idx data to a header list

    Arguments:

    idx: module containing index values

    ignore: list of index values to ignore

    Returns:

    list[str]: ordered list of data array column header names
    """
    idx = globals()[f"idx_{name}"]
    if ignore is None:
        ignore = ignore_idx[name]
    mapping = {getattr(idx,x):x for x in dir(idx) if not x.startswith("_") and x not in ignore}
    indexes = sorted(mapping)
    assert max(indexes) - min(indexes) + 1 == len(indexes), "indexes are not strictly sequential"
    return [mapping[n] for n in indexes]

class PPModel:
    """PyPower Model Access"""

    def __init__(self,
        name:str="unnamed",
        version:int=2,
        mvabase:float=100.0,
        case:dict|Callable=None,
        ):
        """Create PyPower case data

        Arguments:

        name: name of the case

        version: case version number

        mvabase: MVA base value

        case: case data
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
        } if case is None else (case() if callable(case) else case)
        assert "version" in self.case, "version missing in case"
        assert self.case["version"] == 2, f"version={self.case['version']} is not valid"
        assert "baseMVA" in self.case, "baseMVA missing in case"
        assert self.case["baseMVA"] > 0.0, f"baseMVA={self.case['baseMVA']} is not valid"
        assert "bus" in self.case, "bus missing in case"
        assert "branch" in self.case, "branch missing in case"
        assert "gen" in self.case, "gen missing in case"

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
        valid_keys = ["version","baseMVA","bus","branch","gen","gencost","dcline","dclinecost"]
        for key,value in [(x,y) for x,y in self.case.items() if x in valid_keys]:
            if isinstance(value,np.ndarray):
                print(f"""      '{key}': array([""",file=file)
                header = ",".join([f"{{0:>{precision+3}s}}".format(x)
                    for x in get_header(key)])
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
            bus_cols = get_header("bus")
            bus = pd.DataFrame(data=self.case["bus"],
                columns=bus_cols[:self.case["bus"].shape[1]])
            bus.index.name="BUS"
            print(bus,file=file)

        if "branch" in items:
            branch_cols = get_header("branch")
            branch = pd.DataFrame(data=self.case["branch"],
                columns=branch_cols[:self.case["branch"].shape[1]])
            branch.index.name="BRANCH"
            print(branch,file=file)

        if "gen" in items:
            gen_cols = get_header("gen")
            gen = pd.DataFrame(data=self.case["gen"],
                columns=gen_cols[:self.case["gen"].shape[1]])
            gen.index.name="GEN"
            print(gen,file=file)

        if "dcline" in items and "dcline" in self.case and len(self.case["dcline"]) > 0:
            dcline_cols = get_header("dcline")
            dcline = pd.DataFrame(data=self.case["dcline"],
                columns=dcline_cols[:self.case["dcline"].shape[1]])
            dcline.index.name="DCLINE"
            print(dcline,file=file)

        if "gencost" in items:
            cost_cols = get_header("gencost")
            ncost = self.case["gencost"].shape[1] - len(cost_cols)
            cost_cols.extend([f"COST{n}" for n in range(int(ncost))])
            gencost = pd.DataFrame(data=self.case["gencost"],columns=cost_cols)
            gencost.index.name="GENCOST"
            print(gencost,file=file)

        if "dclinecost" in items and "dclinecost" in self.case and len(self.case["dclinecost"]) > 0:
            cost_cols = get_header("dclinecost")
            ncost = self.case["dclinecost"].shape[1] - len(cost_cols)
            cost_cols.extend([f"COST{n}" for n in range(int(ncost))])
            dclinecost = pd.DataFrame(data=self.case["dclinecost"],columns=cost_cols)
            dclinecost.index.name="DCLINECOST"
            print(dclinecost,file=file)

    def save_kml(self,
        filename:str,
        use_geocode:bool=False,
        ):
        """Generate KML output

        Arguments:

        filename: KML filename of output

        use_geocode: marker names are geocode instead of bus id
        """
        kml = KML(filename)

        # bus style
        kml.add_markerstyle(
            name="node",
            url="https://maps.google.com/mapfiles/kml/pal3/icon49.png",
            )

        # bus markers
        for bus_i,latitude,longitude,geocode in self.case["gis"]:
            kml.add_marker(
                name=geocode if use_geocode else f"{bus_i}",
                style="node",
                position=[longitude,latitude,0.0],
                )

        # line style
        kml.add_linestyle(
            name="line-in",
            color="7f00ffff",
            width=4,
            )
        kml.add_linestyle(
            name="line-out",
            color="7f000000",
            width=4,
            )

        # line paths
        gis = {n:(y,x,0,c) for n,x,y,c in self.case["gis"][:,:4]}
        for data in self.case["branch"]:
            fbus = int(data[idx_branch.F_BUS])
            tbus = int(data[idx_branch.T_BUS])
            status = int(data[idx_branch.BR_STATUS])
            kml.add_line(
                name=f"{fbus}-{tbus}",
                style="line-in" if status else "line-out",
                from_position=gis[fbus][0:3],
                to_position=gis[tbus][0:3],
                )
        for data in self.case["dcline"]:
            fbus = int(data[idx_branch.F_BUS])
            tbus = int(data[idx_branch.T_BUS])
            status = int(data[idx_branch.BR_STATUS])
            kml.add_line(
                name=f"{fbus}-{tbus}",
                style="line-in" if status else "line-out",
                from_position=gis[fbus][0:3],
                to_position=gis[tbus][0:3],
                )

        kml.close()

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

        header = get_header("bus")
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
        header = get_header("branch")
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
        for item in get_header("gen"):
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
        for item in get_header("gencost"):
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
        for item in get_header("dcline"):
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
        for item in get_header("dclinecost"):
            if item in kwargs:
                if kwargs[item].ndim == 1:
                    result.append(kwargs[item])
                else:
                    for col in range(kwargs[item].shape[1]):
                        result.append(kwargs[item][:,col])
            else:
                raise ValueError(f"missing {item} data")

        return np.array(result)

    def get_info(self):
        """Get model information"""
        bus = self.get_data("bus")
        gengis = pd.merge(
                self.get_gis(),
                self.get_data("gen"),
                left_on="BUS_I",
                right_on="GEN_BUS")
        loadgis = pd.merge(
                self.get_gis(),
                self.get_data("bus"),
                left_on="BUS_I",
                right_on="BUS_I")
        return {
            "Model name": self.name,
            "Bus count": len(bus),
            "Branch count": len(self.get_data("branch")),
            "Generator count": len(self.get_data("gen")),
            "DC line count": len(self.get_data("dcline")),
            "Node count": len(self.get_nodes()),
            "LV substations": len(bus[bus.BASE_KV==20]),
            "MV substations": len(bus[(bus.BASE_KV>20)&(bus.BASE_KV<250)]),
            "HV substations": len(bus[bus.BASE_KV>250]),
            "Generation substations": len(gengis.GEOHASH.unique()),
            "Load substations": len(loadgis[loadgis.PD>0].GEOHASH.unique()),
            }

    def get_data(self,name) -> pd.DataFrame:
        """Get data table"""
        assert name in ignore_idx, f"'{name}' is not a valid data item name"
        width = self.case[name].shape[1]
        header = get_header(name)
        n = 1
        last = header[-1]
        if len(header) < width:
            header[-1] = f"{last}0"
        while len(header) < width:
            header.append(f"{last}{n}")
            n += 1
        return pd.DataFrame(self.case[name].T,header[:width]).T

    def get_gis(self) -> pd.DataFrame:
        """Get complete GIS data"""
        return pd.DataFrame(self.case["gis"].T,get_header("gis")).T

    def get_nodes(self) -> dict:
        """Get a dictionary of node and their busses"""
        nodes = {}
        for i,j in dict(self.case["gis"][:,[0,3]]).items():
            if j in nodes:
                nodes[j].append(i)
            else:
                nodes[j] = [i]
        return nodes
