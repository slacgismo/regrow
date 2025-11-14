"""PSSE to PyPower converter"""

from ppmodel import PPModel
import pandas as pd
import numpy as np

class PSSE2PP:

    VERBOSE=False
    DEBUG=False

    """PSSE to PyPower converter class"""
    def __init__(self,psse,pp_init={}):
        """Create PSSE to PyPower converter"""

        self.model = PPModel(name=psse.name,**pp_init)
        self.model.case["bus"] = self.bus(psse.bus,psse.load,psse.shunt)

    def bus(self,
        data:pd.DataFrame,
        load:pd.DataFrame,
        shunt:pd.DataFrame
        ) -> np.array:
        """Convert PSSE bus data to PyPower bus data

        Arguments:

        data: bus dataframe from PSSE

        load: load dataframe from PSSE

        shunt: shunt dataframe from PSSE

        Returns:

        np.array: PyPower bus data array
        """
        if self.DEBUG:
            print(f"DEBUG [PSSE2PP]: bus({data=},{load=},{shunt=})")

        load_columns = ["I","PL","QL","IP","IQ","YP","YQ","SCALE","INTRPT","DGENP","DGENQ","DGENF"]
        rawdata = pd.merge(data,load[load_columns],how='left',left_on="ID",right_on="I").fillna(0.0)
        rawdata = pd.merge(rawdata,shunt,how='left',left_on="ID",right_on="I").fillna(0.0)
        rawdata.drop([x for x in rawdata.columns if x.endswith("_x") or x.endswith("_y")],axis=1)
        busdata = self.model.bus(
            BUS_I = rawdata["ID"],
            BUS_TYPE = rawdata["BUSTYPE"],
            PD = rawdata["PL"] + ( rawdata["IP"] + rawdata["YP"] * rawdata["BASEKV"]) * rawdata["BASEKV"],
            QD = rawdata["QL"] + ( rawdata["IQ"] + rawdata["YQ"] * rawdata["BASEKV"] ) * rawdata["BASEKV"],
            GS = np.zeros(len(rawdata)),
            BS = rawdata["BINIT"],
            BUS_AREA = rawdata["AREA"],
            VM = rawdata["VM"],
            VA = rawdata["VA"],
            BASE_KV = rawdata["BASEKV"],
            ZONE = rawdata["ZONE"],
            VMAX = rawdata["VMAX0"],
            VMIN = rawdata["VMIN0"],
        )
        return busdata

    def branch(self,data):
        """Convert PSSE branch data to PyPower bus data"""
        raise NotImplementedError("TODO")

    def gen(self,data):
        """Convert PSSE gen data to PyPower bus data"""
        raise NotImplementedError("TODO")

    def gencost(self,data):
        """Convert PSSE gencost data to PyPower bus data"""
        raise NotImplementedError("TODO")

    def dcline(self,data):
        """Convert PSSE dcline data to PyPower bus data"""
        raise NotImplementedError("TODO")

    def dclinecost(self,data):
        """Convert PSSE dcline data to PyPower bus data"""
        raise NotImplementedError("TODO")
