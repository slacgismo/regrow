"""PSSE model accessor"""
import pandas as pd

class PSSE:
    """PSSE model accessor"""
    VERBOSE=False
    DEBUG=False

    def __init__(self,prefix):
        """Create PSSE model accessor

        Arguments:

        prefix: segment filename prefix, e.g. "wecc240"
        """
        self.name = prefix
        self.area = self.read(f"{prefix}_area.csv",
            converters={
                "ARNAME": str,
            })
        self.bus = self.read(f"{prefix}_bus.csv",
            converters={
                "NAME": str,
            })
        self.branch = self.read(f"{prefix}_branch.csv",
            converters={
                "NAME": str,
            })
        self.gen = self.read(f"{prefix}_gen.csv")
        self.load = self.read(f"{prefix}_load.csv")
        self.shunt = self.read(f"{prefix}_shunt.csv",
            converters={
                "RMIDNT": str,
            })
        self.xform = self.read(f"{prefix}_xform.csv",
            converters={
                "NAME": str,
                "VECGRP": str,
            })
        self.zone = self.read(f"{prefix}_zone.csv",
            converters={
                "ZONAME": str,
            })

    # read the PSSE data tables
    @classmethod
    def read(cls,filename,**kwargs):
        """Read PSSE data segment from file"""
        
        # load segment
        data = pd.read_csv(filename,quotechar="'",**kwargs).fillna(0)

        if cls.VERBOSE:
            print(f"VERBOSE [PSSE]: {filename} is {' rows x '.join([str(x) for x in data.shape])} columns")

        if cls.DEBUG:
            print(f"DEBUG [PSSE]: {filename=}, data=\n{data}")

        return data