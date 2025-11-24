"""Scheduling data accessor

This class reads data from the three scheduling data files

- wecc240_scheduling_generator.csv
- wecc240_scheduling_line.csv
- wecc240_scheduling_storage.csv

These files are manually copied from WECC240_2018_Generation_scheduling.xlsx
and may be edited.

The update_case() method is used to modify a PyPower can to include the data
loaded from scheduling files. Existing generation data is overwritten.
Existing branch status, reactance, and flow limit data is modified using the
line scheduling data. Bus loads are increased using the storage scheduling
data when storage is charging. Otherwise storage is added to generation.
"""

import pandas as pd
import numpy as np
from ppmodel import PPModel
from pypower import idx_cost, idx_bus

class Schedule:
    """Scheduling data accessor class"""

    # pylint: disable=too-few-public-methods

    def __init__(self,
        prefix:str,
        ):

        # create default scheduling data
        self.generator = pd.read_csv(f"{prefix}generator.csv").sort_values("genname")
        self.line = pd.read_csv(f"{prefix}line.csv").sort_values(["StartBusName","EndBusName"])
        self.storage  = pd.read_csv(f"{prefix}storage.csv").sort_values("busname")

    def update_case(self,
        case:dict,
        q_factor:float=10.0,
        init_status:bool=True,
        reset_bustype=False,
        ) -> dict:
        """Update case data from schedule

        Arguments:

        case: case data to udpate

        q_factor: reactive power to use relative to real power

        init_status: flag to override schedule initial status

        reset_bustype: reset all busses to PQ before add new generators

        Return:

        dict: the modified case
        """
        data = self.generator
        mvabase = case["baseMVA"]

        # update the gen data
        case["gen"] = PPModel.gen(
            GEN_BUS = data.busname,
            PG = data.InitPow / mvabase,
            QG = np.zeros(len(data)),
            VG = np.ones(len(data)),
            MBASE=np.full(len(data),mvabase),
            GEN_STATUS=np.ones(len(data)) if init_status else data.InitStatus,
            QMIN = -data.Pmax * q_factor / mvabase,
            QMAX = data.Pmax * q_factor / mvabase,
            PMIN = data.Pmin / mvabase,
            PMAX = data.Pmax / mvabase,
            PC1 = np.zeros(len(data)),
            PC2 = np.zeros(len(data)),
            QC1MIN = np.zeros(len(data)),
            QC1MAX = np.zeros(len(data)),
            QC2MIN = np.zeros(len(data)),
            QC2MAX = np.zeros(len(data)),
            RAMP_AGC = data.Ramp_Rate,
            RAMP_10 = np.zeros(len(data)),
            RAMP_30 = np.zeros(len(data)),
            RAMP_Q = np.zeros(len(data)),
            APF = np.zeros(len(data)),
            ).T

        # update the gencost data
        ncost = [sum(1 if n==0 or x[f"Cost{n+1}"] > 0 else 0 for n in range(4))
            for _,x in data.iterrows()]
        case["gencost"] = PPModel.gencost(
            MODEL = np.ones(len(data),dtype=int),
            STARTUP = data.SUCost,
            SHUTDOWN = data.SDCost,
            NCOST = np.array(ncost).astype(int) + 1,
            COST = np.array([
                np.zeros(len(data)),
                np.zeros(len(data)),
                data.MW1.tolist(),
                data.Cost1.tolist(),
                data.MW2.tolist(),
                data.Cost2.tolist(),
                data.MW3.tolist(),
                data.Cost3.tolist(),
                data.MW4.tolist(),
                data.Cost4.tolist(),
                ]).T
            ).T

        # clear the unused cost gencost columns
        for row in case["gencost"]:
            ncost = row[idx_cost.NCOST]
            if row[idx_cost.MODEL] == idx_cost.PW_LINEAR:
                row[int(idx_cost.COST+ncost*2):] = 0
            elif row[idx_cost.MODEL] == idx_cost.POLYNOMIAL:
                row[int(idx_cost.COST+ncost):] = 0

        # update the branch data
        # TODO: update line data

        # update storage data
        # TODO: not sure how to do that yet

        # update the bus data
        tmp = PPModel("").set_case(case)
        bus = tmp.get_data("bus")
        gen = tmp.get_data("gen")
        ref = bus[bus.BUS_TYPE==idx_bus.REF].index.values # save the reference bus(ses)
        if reset_bustype:
            bus.BUS_TYPE = idx_bus.PQ # change all busses back to PQ
        bus.set_index("BUS_I",inplace=True)
        bus.loc[gen.GEN_BUS,"BUS_TYPE"] = idx_bus.PV # set the gen busses to PV
        bus.reset_index(inplace=True)
        bus.loc[ref,"BUS_TYPE"] = idx_bus.REF # restore the reference bus(ses)
        case["bus"][:,idx_bus.BUS_TYPE] = bus.BUS_TYPE # copy back to case bus data

        # update the gis data
        gis = tmp.get_data("gis")
        gisdata = gis.copy().set_index("BUS_I")

        # # add generator count (NaN -> not allowed)
        gisdata["GEN"] = float('nan') # by default no generation is allowed
        gisdata.loc[bus[bus.BUS_TYPE!=idx_bus.PQ].
            BUS_I,"GEN"] = 0 # all ~PQ busses can have generation
        gisdata.loc[gen.GEN_BUS,"GEN"] = gen.groupby("GEN_BUS")["GEN_STATUS"]\
            .count() # 1 for each gen listed

        # # add load count (NaN -> load not allowed)
        # gisdata["LOAD"] = float('nan') # default no load is allowed
        # gisdata.loc[bus[bus.BUS_TYPE==idx_bus.PQ].BUS_I,"LOAD"] = 0 # all PQ busses can have loads
        # gisdata.loc[bus[bus.PD>0].BUS_I,"LOAD"] = 1 # all load busses have 1 load

        case["gis"] = gisdata.reset_index().values

        return case

if __name__ == "__main__":

    pd.options.display.width = None
    pd.options.display.max_rows = None
    pd.options.display.max_columns = None

    # pylint: disable=cyclic-import,ungrouped-imports
    from wecc240 import wecc240
    from pypower.runpf import runpf
    from pypower.rundcopf import rundcopf
    from pypower.ppoption import ppoption

    casedata = wecc240(options=["SCHEDULING"])

    with open("tests/wecc240_scheduling.py","w",encoding="utf-8") as fh:
        PPModel("wecc240",case=casedata).save_case(fh)

    options = ppoption(VERBOSE=0,OUT_ALL=0)
    assert runpf(casedata,options)[0]["success"], "runpf failed"
    assert rundcopf(casedata,options)["success"], "runopf failed"

    result = Schedule("wecc240/scheduling/").generator
    print(result)
