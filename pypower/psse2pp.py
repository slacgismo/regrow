"""PSSE to PyPower converter"""

from ppmodel import PPModel
import pandas as pd

class PSSE2PP:

    VERBOSE=False
    DEBUG=False

    """PSSE to PyPower converter class"""
    def __init__(self,psse,pp_init={}):
        """Create PSSE to PyPower converter"""

        self.data = PPModel(name=psse.name,**pp_init)
        self.data.case["bus"] = self.bus(psse.bus,psse.load,psse.shunt)

    def bus(self,data,load,shunt):
        """Convert PSSE bus data to PyPower bus data"""
        if self.DEBUG:
            print(f"DEBUG [PSSE2PP]: bus({data=},{load=},{shunt=})")

        BUS_I = data["ID"]
        BUS_TYPE = data["BUSTYPE"]
        BUS_AREA = data["AREA"]
        VM = data["VM"]
        VA = data["VA"]
        BASE_KV = data["BASEKV"]
        ZONE = data["ZONE"]
        VMAX = data["VMAX0"]
        MIN = data["VMIN0"]

        load_columns = ["I","PL","QL","IP","IQ","YP","YQ","SCALE","INTRPT","DGENP","DGENQ","DGENF"]
        rawdata = pd.merge(data,load[load_columns],how='left',left_on="ID",right_on="I").fillna(0.0)
        rawdata = pd.merge(rawdata,shunt,how='left',left_on="ID",right_on="I").fillna(0.0)
        rawdata.drop([x for x in rawdata.columns if x.endswith("_x") or x.endswith("_y")],axis=1)
        print(rawdata)
        # loads = psse.load.set_index("I")
        # shunts = psse.shunt.set_index("I")
        # for n,bus in psse.bus.iterrows():
        #     i = bus["ID"]
        #     load = loads.loc[i] if i in loads.index else None
        #     shunt = shunts.loc[i] if i in shunts.index else None
        #     model.bus(**convert.bus(data=bus,load=load,shunt=shunt))
        busdata = self.data.bus(
            BUS_I = rawdata["ID"],
            BUS_TYPE = rawdata["BUSTYPE"],
            PD = None,
            QD = None,
            GS = None,
            BS = None,
            BUS_AREA = rawdata["AREA"],
            VM = rawdata["VM"],
            VA = rawdata["VA"],
            BASE_KV = rawdata["BASEKV"],
            ZONE = rawdata["ZONE"],
            VMAX = rawdata["VMAX0"],
            MIN = rawdata["VMIN0"],
        )
        return busdata

    def branch(self,data):
        """Convert PSSE branch data to PyPower bus data"""
        return {}

    def gen(self,data):
        """Convert PSSE gen data to PyPower bus data"""
        return {}

    def gencost(self,data):
        """Convert PSSE gencost data to PyPower bus data"""
        return {}

    def dcline(self,data):
        """Convert PSSE dcline data to PyPower bus data"""
        return {}

    def dclinecost(self,data):
        """Convert PSSE dcline data to PyPower bus data"""
        return {}