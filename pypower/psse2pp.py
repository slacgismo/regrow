"""PSSE to PyPower converter

This module defines the class PSSE2PP used to convert a PSSE model to a
PyPower model.

Example:

    from psse import PSSE
    from psse2pp import PSSE2PP
    raw = PSSE(prefix="wecc240/",raw="wecc240_psse.raw")
    ppcase = PSSE2PP(raw).model.case

"""

from typing import TypeVar
import pandas as pd
import numpy as np

from ppmodel import PPModel

# read default costs data (used when no cost data is provided, e.g., from HIFLD)
costs = pd.read_csv("costs.csv",index_col=0)

class PSSE2PP:
    """PSSE to PyPower converter

    Globals:

    LOADSCALE: global load scaling factor (default 1.0)
    """

    LOADSCALE=1.0 # global load scaling

    """PSSE to PyPower converter class"""
    def __init__(self,psse:TypeVar('PPModel')):
        """Create PSSE to PyPower converter

        Arguments:

        psse: PSSE data accessor
        """

        self.mvabase = psse.config["mvabase"]
        self.model = PPModel(name=psse.name,mvabase=self.mvabase)
        self.model.case["bus"] = self.bus(psse.bus,psse.load,psse.shunt)
        self.model.case["gen"] = self.gen(psse.gen)
        self.model.case["gencost"] = self.gencost(psse.gen)
        self.model.case["branch"] = self.branch(psse.branch,psse.xform)
        self.model.case["dcline"] = self.dcline(psse.dcline)
        self.model.case["dclinecost"] = self.dclinecost(psse.dcline)
        self.model.case["gis"] = self.gis(psse.gis)
        self.model.case["scheduling"] = self.scheduling(psse.scheduling)

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

        load_columns = ["I","STAT","SCALE",
            "PL","QL","IP","IQ","YP","YQ",
            "INTRPT","DGENP","DGENQ","DGENF",
            ]
        raw = pd.merge(bus,load[load_columns],how='left',left_on="ID",right_on="I").fillna(0.0)

        shunt_columns = ["I","BINIT","ST"]
        raw = pd.merge(raw,shunt[shunt_columns],how='left',left_on="ID",right_on="I").fillna(0.0)

        p = ( raw["PL"] + raw["IP"] + raw["YP"] ) * raw["STAT"]
        q = ( raw["QL"] + raw["IQ"] - raw["YQ"] ) * raw["STAT"]
        busdata = self.model.bus(
            BUS_I = raw["ID"],
            BUS_TYPE = raw["BUSTYPE"],
            PD = ( p * raw["SCALE"] - raw["DGENP"] ) * self.LOADSCALE / self.mvabase,
            QD = ( q * raw["SCALE"] - raw["DGENQ"] ) * self.LOADSCALE / self.mvabase,
            GS = np.zeros(len(raw)),
            BS = ( raw["BINIT"] * raw["ST"] ) / self.mvabase,
            BUS_AREA = raw["AREA"],
            VM = raw["VM"],
            VA = raw["VA"],
            BASE_KV = raw["BASEKV"],
            ZONE = raw["ZONE"],
            VMAX = raw["VMAX0"],
            VMIN = raw["VMIN0"],
        )
        return np.array(busdata).T

    def gen(self,
        gen:pd.DataFrame,
        ) -> np.array:
        """Convert PSSE gen data to PyPower bus data

        Arguments:

        gen: gen dataframe from PSSE

        Returns:

        np.array: PyPower gen data array
        """

        gendata = self.model.gen(
            GEN_BUS = gen["I"],
            PG = gen["PG"] / self.mvabase,
            QG = gen["QG"] / self.mvabase,
            QMAX = gen["QT"] / self.mvabase,
            QMIN = gen["QB"] / self.mvabase,
            VG = gen["VS"],
            MBASE = gen["MBASE"],
            GEN_STATUS = gen["STAT"],
            PMIN = gen["PB"] / self.mvabase,
            PMAX = gen["PT"] / self.mvabase,
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
        return np.array(gendata).T

    def gencost(self,
        gen:pd.DataFrame,
        ) -> np.array:
        """Convert PSSE gencost data to PyPower bus data

        Arguments:

        gen: gen dataframe from PSSE

        Returns:

        np.array: PyPower gen data array
        """
        costdata = self.model.gencost(
            MODEL = np.array([costs.loc[x].MODEL for x in gen.ID]),
            STARTUP = np.array([costs.loc[x].STARTUP for x in gen.ID]),
            SHUTDOWN = np.array([costs.loc[x].SHUTDOWN for x in gen.ID]),
            NCOST = np.array([costs.loc[x].NCOST for x in gen.ID]),
            COST = np.array([
                [costs.loc[x].COST0 for x in gen.ID],
                [costs.loc[x].COST1 for x in gen.ID],
                [costs.loc[x].COST2 for x in gen.ID],
                ]).T,
            )

        return np.array(costdata).T

    def branch(self,
        branch:pd.DataFrame,
        xform:pd.DataFrame,
        ) -> np.array:
        """Convert PSSE branch data to PyPower bus data

        Arguments:

        branch: branch dataframe from PSSE

        xform: tranformer dataframe from PSSE

        Returns:

        np.array: PyPower branch data array
        """

        linedata = self.model.branch(
            F_BUS = branch["I"],
            T_BUS = branch["J"].abs(), # negative means metered bus (we don't care)
            BR_R = branch["R"],
            BR_X = branch["X"],
            BR_B = branch["B"],
            RATE_A = branch["RATE1"],
            RATE_B = branch["RATE2"],
            RATE_C = branch["RATE3"],
            TAP = np.zeros(len(branch)),
            SHIFT = np.zeros(len(branch)),
            BR_STATUS = branch["STAT"],
            ANGMIN = np.full(len(branch),-360),
            ANGMAX = np.full(len(branch),+360),
        )

        assert (xform["K"]==0).all(), "three-winding transformers not supported"
        xformdata = self.model.branch(
            F_BUS = xform["I"],
            T_BUS = xform["J"],
            BR_R = xform["R12"],
            BR_X = xform["X12"],
            BR_B = xform["MAG2"],
            RATE_A = xform["RATE1_1"],
            RATE_B = xform["RATE1_2"],
            RATE_C = xform["RATE1_3"],
            TAP = xform["WINDV1"],
            SHIFT = xform["ANG1"],
            BR_STATUS = xform["STAT"],
            ANGMIN = np.full(len(xform),-360),
            ANGMAX = np.full(len(xform),+360),
            )

        return np.array(np.hstack([linedata,xformdata])).T

    def dcline(self,
        dcline:pd.DataFrame,
        ) -> np.array:
        """Convert PSSE dcline data to PyPower bus data

        Arguments:

        dcline: dcline dataframe from PSSE

        Returns:

        np.array: PyPower dcline data array
        """

        linedata = self.model.dcline(
            F_BUS = dcline["F_BUS"],
            T_BUS = dcline["T_BUS"],
            BR_STATUS = dcline["BR_STATUS"],
            PF = dcline["PF"],
            PT = dcline["PT"],
            QF = dcline["QF"],
            QT = dcline["QT"],
            VF = dcline["VF"],
            VT = dcline["VT"],
            PMIN = dcline["PMIN"],
            PMAX = dcline["PMAX"],
            QMINF = dcline["QMINF"],
            QMAXF = dcline["QMAXF"],
            QMINT = dcline["QMINT"],
            QMAXT = dcline["QMAXT"],
            LOSS0 = dcline["LOSS0"],
            LOSS1 = dcline["LOSS1"],
        )

        return np.array(linedata).T

    def dclinecost(self,
        dcline:pd.DataFrame,
        ) -> np.array:
        """Convert PSSE dcline data to PyPower bus data

        Arguments:

        dcline: dcline dataframe from PSSE

        Returns:

        np.array: PyPower dclinecost data array
        """

        costdata = self.model.dclinecost(
            MODEL = np.array([costs.loc[x].MODEL for x in dcline.NAME]),
            STARTUP = np.array([costs.loc[x].STARTUP for x in dcline.NAME]),
            SHUTDOWN = np.array([costs.loc[x].SHUTDOWN for x in dcline.NAME]),
            NCOST = np.array([costs.loc[x].NCOST for x in dcline.NAME]),
            COST = np.array([
                [costs.loc[x].COST0 for x in dcline.NAME],
                [costs.loc[x].COST1 for x in dcline.NAME],
                [costs.loc[x].COST2 for x in dcline.NAME],
                ]).T,
            )

        return np.array(costdata).T

    def gis(self,
        gis:pd.DataFrame,
        ) -> list:
        """Convert PSSE gis data to PyPower gis data

        Arguments:

        gis: gis dataframe from PSSE

        Returns:

        list: PyPower gis data
        """

        return gis.values

    def scheduling(self,
        schedule:dict,
        ) -> list:
        """Convert PSSE scheduling data to PyPower scheduling data"""

        return {"future work"}
