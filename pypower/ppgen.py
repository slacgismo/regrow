"""Abstract class for pypower generator data sources"""

import pandas as pd
import numpy as np
from geohash import nearest2
from ppmodel import idx_gis, idx_bus, PPModel

GENDATA = ['state', 'county', 'node', 'bus', 'fuel', 'gen', 'operating_capacity',
       'index', 'variable_cost', 'fixed_cost']

class Generators:
    """Abstract class for generator data"""

    # set of valid columns, data type, and defaults in dataframe
    valid_columns = {
        "state":(str,""),
        "county":(str,""),
        "plant_id":(int,None),
        "generator_id":(str,""),
        "unit_code":(str,""),
        # "owner_id":(str,""),
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

    # allows values for mapping fuel and gen values
    valid_mappings = {
        "fuel": {'WASTE', 'OTHER', 'OIL', 'GAS', 'HYDRO', 'NUCLEAR', 'WIND', 'COAL', 'SOLAR'},
        "gen": {'PV', 'CT', 'NA', 'CC', 'ES', 'WT', 'ST', 'IC', 'HT'},
        }

    def __init__(self,
        source:str=None,
        cache:str=None,
        ):
        """Abstract class constructor for generators

        Arguments:

        source: source of data

        cache: path name to cache
        """

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
        valid_columns = set(self.valid_columns)
        invalid = data_columns - valid_columns
        assert not invalid, f"columns {invalid} are invalid"
        missing = valid_columns - data_columns
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
        q_factor=0.3,
        ignore_bustype:bool=False,
        groupby:list[str]|None=["fuel","gen"],
        converters:dict[dict[str:str]]=None,
        index_csv:str|None=None,
        ) -> pd.DataFrame:
        """Convert generation fleet data to PyPOWER gen data

        Arguments:

        case: pypower case data tables

        ignore_bustype: flag to disable limiting nearest bus search based on bustype

        groupby: data groupings in addition to bus id

        converters: value converters to apply to data columns before groups

        index_csv: CSV file to which gen info is written, same order as gen
        rows, index refers back to data rows
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
        bus_locations = case["gis"][bus_list]
        bus_latlon = [(x[1],x[2]) for x in bus_locations]
        
        # find nearest bus to each generator
        gen_locations = self.data[["latitude","longitude"]].values.tolist()
        gen_bus = [nearest2(xy,bus_latlon)[0] for xy in gen_locations]
        bus_i = bus_locations[gen_bus,idx_gis.BUS_I]

        # map column values
        data = self.data.copy().reset_index()
        data["bus"] = bus_i
        for name,mapping in converters.items() if converters else {}: # apply data converters

            # check if mapping is valid
            if name in self.valid_mappings:
                for value in set(mapping.values()):
                    assert value in self.valid_mappings[name], f"'{value}' is not a valid '{name}' value mapping"
                assert mapping

            # map values
            data[name] = [mapping[x] for x in data[name]]
        data["node"] = [bus_locations[x][idx_gis.GEOHASH] for x in gen_bus]
        data.set_index(["state","county","node","bus","fuel","gen"],inplace=True)

        # aggregation (if any)
        if groupby is None:
            pmax = data["operating_capacity"]
            name = data["index"]
        else:
            groupby = data.groupby(["state","county","node","bus"]+groupby)
            pmax = groupby["operating_capacity"].sum()
            name = groupby["index"].apply(lambda x: ",".join(str(x) for x in set(x)))
            bus_i = pmax.index.get_level_values(3)

        # construct gen data
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

        # save index (to csv if requested)
        gendata = pd.concat([pmax,name],axis=1).round(1).reset_index()
        gendata.index.name = "gen_i"
        gendata[["variable_cost","fixed_cost"]] = 0.0
        if index_csv:
            gendata.to_csv(index_csv,header=True,index=True)
        case["gendata"] = gendata.values

        return result

    def to_gencost(self,
        case:dict,
        costs:pd.DataFrame|None=None
        ):
        """Convert generation fleet data to PyPOWER gencost data

        Arguments:

        case: pypower case data tables

        """
        assert "version" in case and case["version"] == 2, f"{case.version=} is not supported"
        assert "bus" in case, "case must contain bus data"
        assert "gis" in case, "case must contain gis data"
        assert "gendata" in case, "case must contain gendata generated by ppen.to_gen() call"

        # load generation cost data if needed
        if costs is None:
            costs = pd.read_csv("generation_costs.csv",
                index_col=["fuel","gen"],
                usecols=["fuel","gen","variable_cost","fixed_cost"],
                )

        # map generation cost data to gendata
        gendata = pd.DataFrame(case["gendata"],columns=GENDATA)
        gendata.drop(["variable_cost","fixed_cost"],inplace=True,axis=1)
        gendata = pd.merge(left=gendata,right=costs,how="left",
            left_on=["fuel","gen"],right_on=["fuel","gen"],
            )

        result = pd.DataFrame({
            "MODEL": np.full(len(gendata),2.0),
            "STARTUP": np.zeros(len(gendata)),
            "SHUTDOWN": np.zeros(len(gendata)),
            "NCOST": np.full(len(gendata),2.0),
            "COST0": gendata["variable_cost"].values,
            "COST1": gendata["fixed_cost"].values,
            })
        result.fillna(0.0,inplace=True)

        return result 
