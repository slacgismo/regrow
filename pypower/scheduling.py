"""Scheduling data accessor"""

import os
import pandas as pd
import numpy as np
from ppmodel import PPModel

class Schedule:

    def __init__(self,prefix:str):

        for file in [x for x in os.listdir() if x.startswith(prefix) and x.endswith(".csv")]:
            name = file[len(prefix):-len(".csv")]
            setattr(self,name,pd.read_csv(file))

    def update_case(self,
        case:dict,
        Qfactor:float=0.2,
        InitStatus:bool=True,
        ) -> dict:
        """Update case data from schedule

        Arguments:

        case: case data to udpate

        Qfactor: reactive power to use relative to real power

        InitState: flag to override schedule initial status

        Return:

        dict: the modified case
        """
        data = self.generator
        mvabase = case["baseMVA"]
        case["gen"] = PPModel.gen(
            GEN_BUS = data.busname,
            PG = data.InitPow / mvabase,
            QG = np.zeros(len(data)),
            VG = np.ones(len(data)),
            MBASE=np.full(len(data),mvabase),
            GEN_STATUS=np.ones(len(data)) if InitStatus else data.InitStatus,
            QMIN = -data.Pmax * Qfactor / mvabase,
            QMAX = data.Pmax * Qfactor / mvabase,
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
        case["gencost"] = PPModel.gencost(
            MODEL = np.ones(len(data)),
            STARTUP = data.SUCost,
            SHUTDOWN = data.SDCost,
            NCOST = np.full(len(data),4),
            COST = np.array([
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

        return case

if __name__ == "__main__":

    from wecc240 import wecc240
    from pypower.runpf import runpf
    from pypower.rundcopf import rundcopf
    case = wecc240(options=["SCHEDULING"])
    pd.options.display.width = None
    pd.options.display.max_rows = None
    pd.options.display.max_columns = None
    # PPModel("wecc240_schedule").set_case(case).print()
    runpf(case)
