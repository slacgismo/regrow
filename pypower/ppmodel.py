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

Data Structures:

- `case`: Provides all the PyPOWER case data needed to solve powerflow and
  optimal powerflows. See PyPOWER `idx_*` for details.

- `inputs`: Provides all the information required to read data from input
  files and update `case` data.

- `outputs`: Provides all the information required to read `case` data and
  update the output files.

- `options`: Provides all the options used by the PyPOWER solvers. See PyPOWER
  `ppoption` for details.

- `errors`: Records all the error message emitted during a solver call.

- `profile`: Collects all the solver performance data obtained during a solver
  call.

"""

import os
import sys
import io
import datetime as dt
from time import time
from typing import Self, Callable
import warnings

import pytz
import numpy as np
import pandas as pd
from geohash import nearest

from pypower import idx_brch as idx_branch
# pylint: disable=unused-import
from pypower import idx_gen, idx_bus # used indirectly in get_header()
# pylint: enable=unused-import
from pypower import idx_cost as idx_gencost
from pypower.runpf import runpf
from pypower.rundcopf import rundcopf
from pypower.runopf import runopf as runacopf
from pypower.ppoption import ppoption

from kml import KML

idx_dclinecost = idx_gencost

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

standard_idx = {
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
        ignore = standard_idx[name]
    mapping = {getattr(idx,x):x for x in dir(idx) if not x.startswith("_") and x not in ignore}
    indexes = sorted(mapping)
    assert max(indexes) - min(indexes) + 1 == len(indexes), "indexes are not strictly sequential"
    return [mapping[n] for n in indexes]

class PPModel:
    """PyPower Model Access"""

    # pylint: disable=too-many-public-methods

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

        self.inputs = {}
        self.outputs = {}
        self.options = ppoption(VERBOSE=0,OUT_ALL=0)
        self.errors = []
        self.profile = None

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
            items = standard_idx

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
        for bus_i,latitude,longitude,geocode in self.case["gis"][:,0:4]:
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
            "LV busses": len(bus[bus.BASE_KV==20]),
            "MV busses": len(bus[(bus.BASE_KV>20)&(bus.BASE_KV<250)]),
            "HV busses": len(bus[bus.BASE_KV>250]),
            "Generation substations": len(gengis.GEOHASH.unique()),
            "Load substations": len(loadgis[loadgis.PD>0].GEOHASH.unique()),
            }

    def get_data(self,name) -> pd.DataFrame:
        """Get data table"""
        assert name in standard_idx, f"'{name}' is not a valid data item name"
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
        """Get indexed GIS data"""
        return self.get_data("gis").reset_index().sort_index()
        # return pd.DataFrame(self.case["gis"].T,get_header("gis")).T

    def get_loads(self):
        """Get complete load data"""
        return {} # TODO: return full load data

    def get_nodes(self) -> dict:
        """Get a dictionary of node and their busses"""
        nodes = {}
        for i,j in dict(self.case["gis"][:,[0,3]]).items():
            if j in nodes:
                nodes[j].append(i)
            else:
                nodes[j] = [i]
        return nodes

    def get_graph(self,
        level:str="BUS",
        nodes:str=None
        ) -> tuple[pd.DataFrame,pd.DataFrame]:
        """Get network graphs

        Arguments:

        level: "BUS","NODE","ZONE","AREA"

        nodes: return node type (None, "nearest", "centroid")

        Returns:

        links: list of link tuples indexes into bus data in order of branch
        data
        """
        nodes = pd.merge(self.get_data("bus"),
            self.get_data("gis"),
            left_on="BUS_I",
            right_on="BUS_I",
            )

        match level:

            case "BUS":

                # no node aggregation
                nodes["BUS"] = nodes.index # save row indexing for linklist
                nodes.set_index("BUS_I",inplace=True) # index on bus id

                # process branches
                branch = self.get_data("branch").copy()
                branch["FROM"] = nodes.loc[branch.F_BUS.astype(int)].BUS.values
                branch["TO"] = nodes.loc[branch.T_BUS.astype(int)].BUS.values
                branch.set_index(["FROM","TO"],inplace=True)

                # process dclines
                dcline = self.get_data("dcline").copy()
                dcline["FROM"] = nodes.loc[dcline.F_BUS.astype(int)].BUS.values
                dcline["TO"] = nodes.loc[dcline.T_BUS.astype(int)].BUS.values
                dcline.set_index(["FROM","TO"],inplace=True)

                links = branch.index.tolist() + dcline.index.tolist()

            case "GEOHASH":

                warnings.warn(f"{level=} not implemented yet")
                links = pd.DataFrame({"FROM":["-1"],"TO":["-1"]})\
                    .set_index(["FROM","TO"]).index # TODO: write graph at geohash level

            case "ZONE":

                warnings.warn(f"{level=} not implemented yet")
                links = pd.DataFrame({"FROM":["-1"],"TO":["-1"]})\
                    .set_index(["FROM","TO"]).index # TODO: write graph at zone level

            case "AREA":

                warnings.warn(f"{level=} not implemented yet")
                links = pd.DataFrame({"FROM":["-1"],"TO":["-1"]})\
                    .set_index(["FROM","TO"]).index # TODO: write graph at area level

            case "_":

                raise ValueError(f"{level=} is invalid")

        linklist = [[int(y) for y in x] for x in links]
        return linklist

    def map_columns(self,
        name:str,
        column:str,
        lookup:str="gis",
        not_found:str="nearest",
        on_multiple:str="assign",
        basis:str|None=None,
        ):
        """Create a custom mapping for input columns to data rows

        name: name of input target

        column: column of input target

        lookup: source of mapping lookup table

        not_found: handling of columns not found in lookup source

        on_multiple: handling of columns that map to more than one row

        basis: basis GIS column for handling of multiple columns
        """

        # check for and fix missing columns--all should be in gis geohash list)
        gis = self.get_data("gis").copy()
        missing = set(data.columns) - set(gis.GEOHASH)
        match not_found:
            case "nearest":
                geohash_list = gis.GEOHASH.to_list()
                for item in missing:
                    found = nearest(item,geohash_list)
                    data.columns = [found if x==item else x for x in data.columns]
            case "warning":
                for item in missing:
                    warnings.warn(f"{file}: {item} is not in model gis data")
            case "error":
                assert missing == set(), f"{missing} not in GIS data"
            case "_":
                raise ValueError(f"{not_found=} is invalid")

        # map input columns to target rows
        gis.BUS_I = gis.index
        mapping = gis.set_index("GEOHASH").loc[data.columns]
        mapping.index.name="GEOHASH"

        # print(mapping[mapping[basis]>0].reset_index().set_index("BUS_I"))
        result = mapping.loc[data.columns,["BUS_I",basis]]
        noload = result[result["LOAD"].isna()]
        if not noload.empty:
            match not_found:
                case "warning":
                    warnings.warn(f"none of {noload.index} map to load busses")
                case "error":
                    raise KeyError(f"none of {noload.index} map to load busses")
                case "nearest":
                    raise NotImplementedError(f"none of {noload.index} map to load busses;"
                        " {not_found=} is not supported in this case")
                case "_":
                    raise ValueError(f"{not_found=} is invalid")

        self.inputs[(name,column)]["mapping"] = mapping.to_dict()

    def set_input(self,
        name:str,
        column:str,
        file:str,
        scale:float=1.0,
        offset:float=0.0,
        mapping:dict=None,
        # not_found:str="nearest",
        # on_multiple:str="assign",
        # basis:str|None=None,
        ):
        """Set a timeseries input data feed

        Arguments:

        name: data set name (e.g., bus, branch)

        column: data column name (e.g., "PD")

        file: file name from data is input

        scale: scaling factor to apply to the raw data

        offset: offset to apply to the scaled data

        mapping: maps column names to data rows with weights
        """
        assert name in standard_idx,f"{name=} is not valid"
        assert column in get_header(name), f"{column=} is not found in {name} data"
        assert (name,column) not in self.inputs, f"input({name=},{column=}) already defined"
        if file is None:
            del self.inputs[name]
        else:
            assert os.path.exists(file), f"{file=} not found"
            data = pd.read_csv(file,index_col=[0],parse_dates=[0]) * scale + offset
            data.index.name = "datetime"

            # default to direct mapping of column names to row numbers
            if mapping is None:
                mapping = {
                    "index": data.columns.astype(int),
                    "scale": np.ones(len(data.columns)),
                    }

            # set up input
            self.inputs[(name,column)] = {
                "data": data,
                "mapping": mapping,
            }

    def set_output(self,
        name:str,
        column:str,
        file:str,
        scale:float=1.0,
        offset:float=0.0,
        mapping:dict=None,
        format:str="g"):
        """Set a timeseries output data feed"""
        assert name in standard_idx, f"{name=} is not valid"
        assert column in get_header(name), f"{column=} is not found in {name} data"
        assert file not in self.outputs, f"{file=} already exists in the outputs"
        if mapping is None:
            nrows = len(self.case[name])
            mapping = {
                "rows": np.arange(nrows,dtype=int),
                "columns": [f"{x}" for x in range(nrows)],
                "scale": np.full(nrows,scale),
                "offset": np.full(nrows,offset),
            }
        fh = open(file,"w",encoding="utf-8")
        self.outputs[file] = {
            "name": name,
            "column": column,
            "fh":fh,
            "mapping": mapping,
            "format": format,
            }
        print("datetime",*mapping["columns"],sep=",",file=fh)

    def run_timeseries(self,*args,
        # pylint: disable=too-many-arguments,too-many-locals
        progress:Callable=None,
        call_on_fail:Callable=None,
        stop_on_fail:bool=True,
        stop_test:Callable=None,
        use_acopf:bool=False,
        **kwargs):
        """Run a timeseries simulation

        Arguments:

        *args, **kwargs: See pandas.date_range()

        progress: set a progress callback function

        call_on_fail: set a call-on-fail function

        stop_on_fail: enable stop-on-fail condition

        stop_test: set a stop test call back function

        use_acopf: enable use of AC OPF instead of DC OPF

        Returns:

        None: No errors to report

        str: Error message (when stop_on_fail is True)

        list[str]: Error messages (when stop_on_fail is False)
        """

        self.errors = [] # collect errors, if any
        tic0 = time()

        # process time specified range
        trange = pd.date_range(*args,**kwargs)
        niters = 0
        topf = 0.0
        tpf = 0.0
        for t in (x.tz_convert("UTC") for x in trange):

            # setup time and progress/stop callback
            ts = t.strftime("%Y-%m-%d %H:%M:%S %Z")
            if callable(progress) and progress(f"""{ts} ({len(self.errors)
                    if self.errors else 'no'} errors)"""):
                return None

            # update inputs
            for name,spec in self.inputs.items():
                data = spec["data"]
                name,column = name
                column_number = get_header(name).index(column)
                mapping = spec["mapping"]["index"]
                scales = spec["mapping"]["scale"]
                try:
                    target = self.case[name]
                    target[mapping,column_number] = data.loc[t] * scales
                except KeyError as exception:
                    warnings.warn(f"input({name=},{column=}) {exception=}")

            # solve OPF and check result
            tic1 = time()
            opf = self.solve_opf(use_acopf)
            toc1 = time()
            if opf["success"] != 1:
                failed = f"OPF failed at {ts}"
                self.errors.append(failed)
                if call_on_fail:
                    call_on_fail(failed)
                if stop_on_fail:
                    break
            topf += toc1 - tic1

            # solver powerflow and check result
            tic1 = time()
            _,status = self.solve_pf()
            toc1 = time()
            if status != 1:
                failed = f"PF failed at {ts}"
                self.errors.append(failed)
                if call_on_fail:
                    call_on_fail(failed)
                if stop_on_fail:
                    break
            tpf += toc1 - tic1

            # process outputs
            for file,spec in self.outputs.items():
                mapping = spec["mapping"]
                scale = mapping["scale"]
                offset = mapping["offset"]
                formt = spec["format"]
                data = [f"{{0:{formt}}}".format(x)
                    for x in self.get_data(
                        spec["name"]).loc[mapping["rows"],spec["column"]]*scale + offset]
                print(ts,*data,sep=",",file=spec["fh"],flush=True)

            # check stop condition
            niters += 1
            if stop_test and stop_test(t):
                self.errors.append(f"Stopped at {t=}")
                break

        ttot = time() - tic0
        self.profile = {
            "Expected iterations": len(trange),
            "Completed iterations": niters,
            "Total OPF time (s)": round(topf,4),
            "Fraction OPF time (s/s)": round(topf/ttot,2) if ttot > 0 else 0,
            "Total PF time (s)": round(tpf,4),
            "Fraction PF time (s/s)": round(tpf/ttot,2) if ttot > 0 else 0,
            "Total run time (s)": round(ttot,4),
            "Iteration time (s/iter)": round(ttot/niters,4) if niters > 0 else "N/A",
        }

        return self.errors if self.errors else None

    def set_datetime(self,datetime):
        """Set the model date/time"""

    def solve_pf(self):
        """Solve the powerflow problem"""
        return runpf(self.case,self.options)

    def solve_opf(self,use_acopf:bool=False):
        """Solve the optimal powerflow problem"""
        return (runacopf if use_acopf else rundcopf)(self.case,self.options)

if __name__ == "__main__":

    pd.options.display.width = None
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None

    from wecc240 import wecc240
    options = ["SCHEDULING"]
    model = PPModel(case=wecc240(options=options))

    # print(model.get_gis())
    # quit()
    # print(model.get_data("gis"))

    for graph in ["BUS","GEOHASH","ZONE","AREA"]:
        print(f"{graph}:",model.get_graph(graph))

    model.set_input("bus","PD","tests/load.csv",scale=10)
    model.set_input("bus","QD","tests/load.csv",scale=1)

    model.set_output("bus","VM","results/bus_vm.csv",format=".3f")
    model.set_output("bus","VA","results/bus_va.csv",format=".4f")
    model.set_output("bus","PD","results/bus_pd.csv",format=".4f")
    model.set_output("bus","QD","results/bus_qd.csv",format=".4f")

    SIM_RESULT =  model.run_timeseries(
        "2020-08-01 00:00:00+07:00",
        "2020-09-01 00:00:00+07:00",
        freq="1h",
        progress=lambda x: print(x,flush=True),
        stop_test=lambda x: x > dt.datetime(2020,8,1,0,0,0,tzinfo=pytz.UTC),
        use_acopf=False,
        )
    assert SIM_RESULT == [
        "Stopped at t=Timestamp('2020-08-01 01:00:00+0000', tz='UTC')",
        ], f"ERROR: {SIM_RESULT}"
    print("Simulation test ok")
    print(model.profile)
