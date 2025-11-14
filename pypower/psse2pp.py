"""PSSE to PyPower converter

This module defines the class PSSE2PP used to convert a PSSE model to a
PyPower model.

Example:

    from psse import PSSE
    from psse2pp import PSSE2PP
    raw = PSSE("wecc240")
    ppcase = PSSE2PP(raw).model.case

"""

from ppmodel import PPModel
import pandas as pd
import numpy as np
import defaults

from typing import TypeVar

class PSSE2PP:
    """PSSE to PyPower converter

    Globals:

    DEBUG: enable debug output (default False)
    LOADSCALE: global load scaling factor (default 1.0)
    VERBOSE: enable verbose output (default False)
    """

    VERBOSE=False
    DEBUG=False
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

        PD = raw["PL"] + raw["IP"] + raw["YP"]
        QD = raw["QL"] + raw["IQ"] - raw["YQ"]
        busdata = self.model.bus(
            BUS_I = raw["ID"],
            BUS_TYPE = raw["BUSTYPE"],
            PD = ( PD * raw["SCALE"] - raw["DGENP"] ) * self.LOADSCALE / self.mvabase,
            QD = ( QD * raw["SCALE"] - raw["DGENQ"] ) * self.LOADSCALE / self.mvabase,
            GS = np.zeros(len(raw)),
            BS = ( raw["BINIT"] * raw["ST"] * raw["N1"] ) / self.mvabase,
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

        if self.DEBUG:
            print(f"DEBUG [PSSE2PP]: gen({gen=})")

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
        """Convert PSSE gencost data to PyPower bus data"""
        if self.DEBUG:
            print(f"DEBUG [PSSE2PP]: gen({gen=})")

        costs = np.array([defaults.gencost[x] for x in gen["ID"]]).T
        costdata = self.model.gencost(
            MODEL = costs[0],
            STARTUP = costs[1],
            SHUTDOWN = costs[2],
            NCOST = costs[3],
            COST = costs[4:].T,
            )

        return np.array(costdata).T

    def branch(self,
        branch:pd.DataFrame,
        xform:pd.DataFrame,
        ) -> np.array:
        """Convert PSSE branch data to PyPower bus data"""

        if self.DEBUG:
            print(f"DEBUG [PSSE2PP]: branch({branch=},{xform=})")

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
        load:pd.DataFrame,
        gen:pd.DataFrame) -> np.array:
        """Convert PSSE dcline data to PyPower bus data"""
        raise NotImplementedError("TODO")

    def dclinecost(self,
        load:pd.DataFrame,
        gen:pd.DataFrame,
        ) -> np.array:
        """Convert PSSE dcline data to PyPower bus data"""
        raise NotImplementedError("TODO")
