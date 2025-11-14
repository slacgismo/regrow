"""PSSE model accessor"""
import pandas as pd

class PSSE:
    """PSSE model accessor"""
    
    VERBOSE=False
    DEBUG=False

    def __init__(self,prefix:str):
        """Create PSSE model accessor

        Arguments:

        prefix: segment filename prefix, e.g. "wecc240"
        """

        # read and check the model config data (first row, second and third columns)
        self.config = dict(zip(
            ["mvabase","version"],
            pd.read_csv(f"{prefix}_psse.raw",nrows=1,usecols=range(1,3),header=None).loc[0].tolist()
            ))
        assert self.config["version"] == 34, f"PSS/E version {self.config['version']} not"
        assert self.config["mvabase"] > 0, f"PSS/E MVA base must be positive"
        
        # save the prefix as the default model name (changing it later is ok)
        self.name = prefix

        # load the segment files
        self.area = self.read(f"{prefix}_area.csv",
            converters={
                "ARNAME": str,
            })
        self.bus = self.read(f"{prefix}_bus.csv",
            converters={
                "NAME": str,
                "BUSTYPE": float,
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
    def read(cls,
        filename:str,
        **kwargs,
        ) -> pd.DataFrame:
        """Read PSSE data segment from file

        Arguments:

        filename: PSSE segment file name

        kwargs: pd.read_csv(**kwargs)

        Returns:

        pd.DataFrame: data frame containing PSSE segment data
        """
        
        if cls.VERBOSE:
            print(f"VERBOSE [PSSE]: {filename} is {' rows x '.join([str(x) for x in data.shape])} columns")

        # load segment and clean up quotes and NaNs
        data = pd.read_csv(filename,quotechar="'",**kwargs).fillna(0)

        if cls.DEBUG:
            print(f"DEBUG [PSSE]: {filename=}, data=\n{data}")

        return data
