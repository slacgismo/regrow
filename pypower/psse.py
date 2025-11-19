"""PSSE model accessor

This module defines the PSSE model access class. Note that you must manually
extract the segments from the PSSE RAW file, cleanup the header line, and
remove extraneous whitespaces from inside strings. The convention is that the
segment files share the same prefix as the original RAW file, but they are CSV
files with the "psse", replaced by "area", "branch", "bus", "gen", "load", "shunt",
"xform", and "zone", which are all required. Fortunately, this is a one-time
 task and it can be automated (someday) if necessary.

The class also reads all the auxiliary data for GIS, scheduling, HIFLD, load
modeling, and renewables that can be used to assemble a PyPower case.

Example:

    from psse import PSSE
    raw = PSSE(prefix="wecc240/",raw="wecc240_psse.raw")
    print(raw.bus)
"""

import pandas as pd
from geohash import geohash

class PSSE:
    """PSSE model accessor class"""

    # pylint: disable=too-many-instance-attributes, too-few-public-methods

    def __init__(self,
        prefix:str,
        raw:str,
        ):
        """Create PSSE model accessor

        Arguments:

        prefix: segment filename prefix, e.g. "wecc240/"

        raw: PSS/E raw filename
        """

        # read and check the model config data (first row, second and third columns)
        self.config = dict(zip(
            ["mvabase","version"],
            pd.read_csv(raw,nrows=1,usecols=range(1,3),header=None).loc[0].tolist()
            ))
        assert self.config["version"] == 34, f"PSS/E version {self.config['version']} not"
        assert self.config["mvabase"] > 0, "PSS/E MVA base must be positive"

        # save the prefix as the default model name (changing it later is ok)
        self.name = prefix

        # load the segment files
        self.area = self.read(f"{prefix}area.csv",
            converters={
                "ARNAME": str,
            })
        self.bus = self.read(f"{prefix}bus.csv",
            converters={
                "NAME": str,
                "BUSTYPE": float,
            })
        self.branch = self.read(f"{prefix}branch.csv",
            converters={
                "NAME": str,
            })
        self.gen = self.read(f"{prefix}gen.csv")
        self.load = self.read(f"{prefix}load.csv")
        self.shunt = self.read(f"{prefix}shunt.csv",
            converters={
                "RMIDNT": str,
            })
        self.xform = self.read(f"{prefix}xform.csv",
            converters={
                "NAME": str,
                "VECGRP": str,
            })
        self.zone = self.read(f"{prefix}zone.csv",
            converters={
                "ZONAME": str,
            })
        self.dcline = self.read(f"{prefix}dcline.csv",
            converters={"NAME":str})

        self.gis = self.read(f"{prefix}gis.csv")

        # geohash missing?
        if "GEOHASH" not in self.gis.columns:

            # add geohash
            self.gis["GEOHASH"] = [geohash(x,y) for x,y in zip(self.gis.LAT,self.gis.LON)]

            # save back to original file for others to use
            self.gis.to_csv(f"{prefix}gis.csv",index=False,header=True)

        self.scheduling = {x:self.read(f"{prefix}scheduling/{x}.csv")
            for x in ["generator","line","storage"]}

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

        # load segment and clean up quotes and NaNs
        return pd.read_csv(filename,
            quotechar="'",
            comment="#",
            **kwargs).fillna(0)

if __name__ == "__main__":

    data = PSSE("wecc240/","wecc240_psse.raw")
    print(data.bus.groupby("BASEKV").count()["ID"])
    print(data.bus[data.bus.BASEKV==20]["ID"].tolist())
