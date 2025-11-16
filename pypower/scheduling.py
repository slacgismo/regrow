"""Scheduling data accessor"""

import os
import pandas as pd
import numpy as np
from ppmodel import PPModel

class Schedule:
    """Scheduling data accessor class"""

    # pylint: disable=too-few-public-methods

    def __init__(self,prefix:str):

        # create default scheduling data
        self.generator = None
        self.line = None
        self.storage = None

        # read scheduling data from CSV files
        for file in [x for x in os.listdir() if x.startswith(prefix) and x.endswith(".csv")]:
            name = file[len(prefix):-len(".csv")]
            setattr(self,name,pd.read_csv(file))

    def update_case(self,
        case:dict,
        q_factor:float=0.2,
        init_status:bool=True,
        ) -> dict:
        """Update case data from schedule

        Arguments:

        case: case data to udpate

        q_factor: reactive power to use relative to real power

        init_status: flag to override schedule initial status

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
        case["gencost"] = PPModel.gencost(
            MODEL = np.ones(len(data)),
            STARTUP = data.SUCost,
            SHUTDOWN = data.SDCost,
            NCOST = np.full(len(data),8),
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

    # pylint: disable=cyclic-import
    from wecc240 import wecc240
    from pypower.runpf import runpf

    casedata = wecc240(options=["SCHEDULING"])

    pd.options.display.width = None
    pd.options.display.max_rows = None
    pd.options.display.max_columns = None

    runpf(casedata)

    PPModel("wecc240").set_case(casedata).print()
