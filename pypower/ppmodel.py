"""PyPower model accessor"""

from pypower import idx_bus
from pypower import idx_brch
from pypower import idx_gen

class PPModel:
    """PyPower Model Access"""

    VERBOSE=False
    DEBUG=False

    def __init__(self,name,version=2,mvabase=100.0):
        if self.VERBOSE:
            print(f"VERBOSE [PPModel]: creating model {name=} {version=} {mvabase=}")
        self.name = name
        self.case = {
            "version": version,
            "mvabase": mvabase,
            "bus": [],
            "branch": [],
            "gen" : [],
            "gencost": [],
            "dcline": [],
            "dclinecost": [],
        }

    def bus(self,**kwargs):
        """Create bus data"""
        if self.DEBUG:
            print(f"DEBUG [PPModel]: add bus data={kwargs}")
        return TODO

    def branch(self,**kwargs):
        """Create branch data"""
        pass

    def gen(self,**kwargs):
        """Create gen data"""
        pass

    def gencost(self,**kwargs):
        """Create gencost data"""
        pass

    def dcline(self,**kwargs):
        """Create dcline data"""
        pass

    def dclinecost(self,**kwargs):
        """Create dclinecost data"""
        pass