"""WECC240 case converter from PSS/E

This converter is designed to be used the same way a case file is used
in PyPOWER, i.e.,

    from pypower.runpf import runpf
    from wecc240 import wecc240
    case = wecc240()
    runpf(case)
"""

import pandas as pd

from pypower import idx_bus
from pypower import idx_brch
from pypower import idx_gen


class PSSE:

    VERBOSE=False

    def __init__(self,prefix):

        self.area = self.read(f"{prefix}_area.csv",index_col=[0])
        self.bus = self.read(f"{prefix}_bus.csv",index_col=[0])
        self.branch = self.read(f"{prefix}_branch.csv",index_col=[0,1,2])
        self.gen = self.read(f"{prefix}_gen.csv",index_col=[0,1])
        self.load = self.read(f"{prefix}_load.csv",index_col=[0,1])
        self.shunt = self.read(f"{prefix}_shunt.csv",
            index_col=[0],
            converters={
                "RMIDNT": str,
            })
        self.xform = self.read(f"{prefix}_xform.csv",
            index_col=[0,1,2,3],
            converters={
                "NAME": str,
                "VECGRP": str,
            }
            )
        self.zone = self.read(f"{prefix}_zone.csv")

    # read the PSSE data tables
    @classmethod
    def read(cls,filename,**kwargs):
        """Read PSSE data segment from file"""
        
        # load segment
        data = pd.read_csv(filename,quotechar="'",**kwargs)

        if cls.VERBOSE:
            print(f"VERBOSE: {filename} is {' rows x '.join([str(x) for x in data.shape])} columns")

        return data

def wecc240():
    """Load and convert the WECC 240 PSSE model to a PyPOWER case"""

    data = {
        "version" : 2,
        "baseMVA" : 100.0,
    }
    psse = PSSE("wecc240")

    # TODO: create pypower bus, branch, and gen data
    
    return data

if __name__ == "__main__":

    pd.options.display.max_columns = None
    pd.options.display.width = None
    pd.options.display.max_rows = None

    from pypower.runpf import runpf
    case = wecc240()
    # runpf(case)