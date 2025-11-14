"""PSSE to PyPower converter"""

from ppmodel import PPModel
import pandas as pd
import numpy as np
import defaults

class PSSE2PP:

    VERBOSE=False
    DEBUG=False

    """PSSE to PyPower converter class"""
    def __init__(self,psse,pp_init={}):
        """Create PSSE to PyPower converter"""

        self.model = PPModel(name=psse.name,**pp_init)
        self.model.case["bus"] = self.bus(psse.bus,psse.load,psse.shunt)
        self.model.case["gen"] = self.gen(psse.gen)
        self.model.case["gencost"] = self.gencost(psse.gen)
        # self.model.case["branch"] = self.branch(psse.branch,psse.xform)
        # self.model.case["dcline"] = self.dcline(psse.load,psse.gen)
        # self.model.case["dclinecost"] = self.dclinecost(psse.load,psse.gen)

    def bus(self,
        bus:pd.DataFrame,
        load:pd.DataFrame,
        shunt:pd.DataFrame
        ) -> np.array:
        """Convert PSSE bus data to PyPower bus data

        Arguments:

        bus: bus dataframe from PSSE

        load: load dataframe from PSSE

        shunt: shunt dataframe from PSSE

        Returns:

        np.array: PyPower bus data array
        """

        if self.DEBUG:
            print(f"DEBUG [PSSE2PP]: bus({bus=},{load=},{shunt=})")

        load_columns = ["I","PL","QL","IP","IQ","YP","YQ","SCALE","INTRPT","DGENP","DGENQ","DGENF"]
        raw = pd.merge(bus,load[load_columns],how='left',left_on="ID",right_on="I").fillna(0.0)
        raw = pd.merge(raw,shunt,how='left',left_on="ID",right_on="I").fillna(0.0)
        raw.drop([x for x in raw.columns if x.endswith("_x") or x.endswith("_y")],axis=1)
        busdata = self.model.bus(
            BUS_I = raw["ID"],
            BUS_TYPE = raw["BUSTYPE"],
            PD = raw["PL"] + ( raw["IP"] + raw["YP"] * raw["BASEKV"]) * raw["BASEKV"],
            QD = raw["QL"] + ( raw["IQ"] + raw["YQ"] * raw["BASEKV"] ) * raw["BASEKV"],
            GS = np.zeros(len(raw)),
            BS = raw["BINIT"],
            BUS_AREA = raw["AREA"],
            VM = raw["VM"],
            VA = raw["VA"],
            BASE_KV = raw["BASEKV"],
            ZONE = raw["ZONE"],
            VMAX = raw["VMAX0"],
            VMIN = raw["VMIN0"],
        )
        return busdata

    def gen(self,gen):
        """Convert PSSE gen data to PyPower bus data

        Arguments:

        gen: gen dataframe from PSSE

        Returns:

        np.array: PyPower gen data array
        """

        if self.DEBUG:
            print(f"DEBUG [PSSE2PP]: gen({gen=})")

        gendata = self.model.gen(
            GEN_BUS = gen["I"],
            PG = gen["PG"],
            QG = gen["QG"],
            QMAX = gen["QT"],
            QMIN = gen["QB"],
            VG = gen["VS"],
            MBASE = gen["MBASE"],
            GEN_STATUS = gen["STAT"],
            PMIN = gen["PB"],
            PMAX = gen["PT"],
            PC1 = np.zeros(len(gen)),
            PC2 = np.zeros(len(gen)),
            QC1MIN = np.zeros(len(gen)),
            QC1MAX = np.zeros(len(gen)),
            QC2MIN = np.zeros(len(gen)),
            QC2MAX = np.zeros(len(gen)),
            RAMP_AGC = np.zeros(len(gen)),
            RAMP_10 = np.zeros(len(gen)),
            RAMP_30 = np.zeros(len(gen)),
            RAMP_Q = np.zeros(len(gen)),
            APF = np.zeros(len(gen)),
            )
        return np.array(gendata)

    def gencost(self,gen):
        """Convert PSSE gencost data to PyPower bus data"""
        if self.DEBUG:
            print(f"DEBUG [PSSE2PP]: gen({gen=})")

        gencost = [defaults.gencost[x] for x in gen["ID"]]

        return np.array(gencost)

    def branch(self,branch,xform):
        """Convert PSSE branch data to PyPower bus data"""

        if self.DEBUG:
            print(f"DEBUG [PSSE2PP]: branch({branch=},{xform=})")

        raise NotImplementedError("TODO")

    def dcline(self,load,gen):
        """Convert PSSE dcline data to PyPower bus data"""
        raise NotImplementedError("TODO")

    def dclinecost(self,load,gen):
        """Convert PSSE dcline data to PyPower bus data"""
        raise NotImplementedError("TODO")
