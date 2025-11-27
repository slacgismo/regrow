"""Abstract class for pypower generator data sources"""

import pandas as pd
import numpy as np
from geohash import nearest2
from ppmodel import idx_gis, idx_bus, PPModel

class Generators:
    """Abstract class for generator data"""

    # set of valid columns, data type, and defaults in dataframe
    valid_columns = {
        "state":(str,""),
        "county":(str,""),
        "plant_id":(int,None),
        "generator_id":(str,""),
        "unit_code":(str,""),
        "owner_id":(str,""),
        "plant_name":(str,""),
        "operating_capacity":(float,float('nan')),
        "summer_capacity":(float,float('nan')),
        "winter_capacity":(float,float('nan')),
        "technology":(str,""),
        "fuel":(str,""),
        "gen":(str,""),
        "latitude":(float,float('nan')),
        "longitude":(float,float('nan')),
        "geohash":(str,""),
    }

    def __init__(self,
        source:str=None,
        cache:str=None):
        """Abstract class constructor for generators"""

        # verify source and cache specs
        assert isinstance(source,str), "source is not a valid string"
        assert isinstance(cache,str), "cache is not a valid string"
        self.source=source
        self.cache=cache

        # check data type
        if not hasattr(self,"data"):
            self.data = None

        # verify data is a valid dataframe
        assert hasattr(self,"data"), "concrete class missing data attribute"
        assert isinstance(self.data,pd.DataFrame), "data is not a Pandas dataframe"

        # verify columns match valid columns
        data_columns = set(self.data.columns)
        invalid = data_columns - set(self.valid_columns)
        missing = set(self.valid_columns) - data_columns
        assert not invalid, f"columns {invalid} are invalid"
        assert not missing, f"columns {missing} are missing"

        # correct column data with data types and defaults from valid_columns
        for name,spec in self.valid_columns.items():
            def convert(x,dtype,default):
                try:
                    return dtype(x)
                except:
                    return default
            self.data[name] = [convert(x,*spec) for x in self.data[name]]

    def to_gen(self,
        case:dict,
        q_factor=1.0,
        ignore_bustype:bool=False,
        ) -> pd.DataFrame:
        """Convert generation fleet data to PyPOWER gen data

        Arguments:

        case: pypower case data table

        ignore_bustype: flag to disable limiting nearest bus search based on bustype
        """
        assert "version" in case and case["version"] == 2, f"{case.version=} is not supported"
        assert "bus" in case, "case must contain bus data"
        assert "gis" in case, "case must contain gis data"
        assert isinstance(q_factor,float) and q_factor >= 0.0, f"{q_factor=} is not valid"
        assert isinstance(ignore_bustype,bool), f"{ignore_bustype=} is not valid"

        # generation types
        gen_types = self.data.set_index(["fuel","gen","plant_id"])
        capacities = gen_types.groupby("plant_id")[
            ["operating_capacity","summer_capacity","winter_capacity"]
            ].sum()
        counts = gen_types.groupby("plant_id")["plant_name"].count()
        gen_data = pd.merge(capacities,counts,left_on=capacities.index.names,right_on=counts.index.names)

        # get list of acceptable busses we can map gens to
        if ignore_bustype == True:
            bus_list = None # index all
        else:
            bus_list = case["bus"][:,idx_bus.BUS_TYPE].astype(int)
            bus_list = [n for n,x in enumerate(bus_list) if x != idx_bus.PQ]
        bus_locations = case["gis"][bus_list]#,np.s_[idx_gis.LAT,idx_gis.LON]]
        bus_latlon = [(x[1],x[2]) for x in bus_locations]
        
        # find nearest bus to each generator
        gen_locations = self.data[["latitude","longitude"]].values.tolist()
        gen_bus = [nearest2(xy,bus_latlon)[0] for xy in gen_locations]
        bus_i = bus_locations[gen_bus,idx_gis.BUS_I]
        
        pmax = self.data.operating_capacity
        result = pd.DataFrame({
            "GEN_BUS": bus_i,
            "PG": np.zeros(len(bus_i)),
            "QG": np.zeros(len(bus_i)),
            "QMAX": pmax * q_factor,
            "QMIN": -pmax * q_factor,
            "VG": np.ones(len(bus_i)),
            "MBASE": np.full(len(bus_i),case["baseMVA"]),
            "GEN_STATUS": np.ones(len(bus_i)),
            "PMAX": pmax,
            "PMIN": np.zeros(len(bus_i)),
            "PC1": np.zeros(len(bus_i)),
            "PC2": np.zeros(len(bus_i)),
            "QC1MIN": np.zeros(len(bus_i)),
            "QC1MAX": np.zeros(len(bus_i)),
            "QC2MIN": np.zeros(len(bus_i)),
            "QC2MAX": np.zeros(len(bus_i)),
            "RAMP_AGC": np.zeros(len(bus_i)),
            "RAMP_10": np.zeros(len(bus_i)),
            "RAMP_30": np.zeros(len(bus_i)),
            "RAMP_Q": np.zeros(len(bus_i)),
            "APF": np.zeros(len(bus_i)),
            })
        return result

    def to_gencost(self):
        """Convert generation fleet data to PyPOWER gencost data"""
        return {} # TODO: create gencost data table
