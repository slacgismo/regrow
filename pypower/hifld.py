"""This module implements the HIFLD data accessor class

The `HIFLD` class loads the HIFLD powerplant data stored in powerplants.csv.zip.

The `drop_test` option can be used to remove unwanted records from the database
after it is loaded. This is needed to remove non-WECC powerplants and
non-operational units, e.g.,

    drop_test=lambda x: x[(~x["STATE"].isin(WECC))|(x["STATUS"]!="OP")].index,

The `drop_na` option removes all records that have NA plant capacity values
(these are presumably unusable).

The `drop_types` and `drop_fuels` options remove all powerplants that have the
specified plant type and fuel codes, respectively.  See the `PLANT_TYPES`.

The `busdata` dataframe is used to restrict the list of busses from which location
are chosen if `geohash_precision` is specified.

The `groupby` list is used to group capacities by the specified column groups. For
example `["BUSCODE","TYPE"]` will group the plants by bus geocode and generation type
and return the total operating, winter, and summer capacities.

Example:

The following loads all operational units in WECC except PV and WT units.

    data = HIFLD(
        drop_test=lambda x: x[(~x["STATE"].isin(WECC))|(x["STATUS"]!="OP")].index,
        drop_types=["PV","WT"],
        drop_fuels=["SUN","WIND"],
        )

The following example limits the buslist to only those with 20kV substations and 
group the plants by bus:

    gis = pd.read_csv("wecc240/gis.csv")
    bus = pd.read_csv("wecc240/bus.csv")
    busses = pd.merge(gis,bus,left_on="BUS_I",right_on="ID").drop("ID",axis=1)
    busses.drop(busses[busses.BASEKV>30.0].index,inplace=True)
    data = HIFLD(
        drop_test=lambda x: x[(~x["STATE"].isin(WECC))|(x["STATUS"]!="OP")].index,
        busdata=busses,
        groupby=["BUSCODE"]
        )

"""

import pandas as pd
from geohash import geohash, nearest2

PLANT_TYPES = {
    # energy source and power generator
    'COAL INTEGRATED GASIFICATION COMBINED CYCLE' : ["COAL","CC"],
    'NATURAL GAS FIRED COMBINED CYCLE' : ["GAS", "CC"],
    'NATURAL GAS FIRED COMBUSTION TURBINE' : ["GAS","CT"],
    'PETROLEUM COKE' : ["COKE","ST"],
    'NATURAL GAS INTERNAL COMBUSTION ENGINE' : ["GAS","IC"],
    'NATURAL GAS STEAM TURBINE' : ["GAS","ST"],
    'PETROLEUM LIQUIDS' : ["OIL","ST"],
    'OTHER GASES' : ["GAS","ST"],
    'CONVENTIONAL STEAM COAL' : ["COAL","ST"],
    'SOLAR PHOTOVOLTAIC' : ["SUN","PV"],
    'BATTERIES' : ["ELEC","ES"],
    'MUNICIPAL SOLID WASTE' : ["WASTE","CT"],
    'NUCLEAR' : ["NUC","ST"],
    'OTHER WASTE BIOMASS' : ["BIO","ST"],
    'ONSHORE WIND TURBINE' : ["WIND","WT"],
    'CONVENTIONAL HYDROELECTRIC' : ["WATER","HT"],
    'WOOD/WOOD WASTE BIOMASS' : ["WOOD","ST"],
    'FLYWHEELS' : ["ELEC","FW"],
    'GEOTHERMAL' : ["GEO","ST"],
    'OTHER NATURAL GAS' : ["GAS","ST"],
    'LANDFILL GAS' : ["WASTE","CT"],
    'ALL OTHER' : ["OTHER","UNKNOWN"],
    'NATURAL GAS WITH COMPRESSED AIR STORAGE' : ["ELEC","AT"],
    'HYDROELECTRIC PUMPED STORAGE' : ["ELEC","HT"],
    'SOLAR THERMAL WITHOUT ENERGY STORAGE' : ["SUN","ST"],
    'SOLAR THERMAL WITH ENERGY STORAGE' : ["SUN","ES"],
    'NOT AVAILABLE' : ["UNKNOWN","UNKNOWN"],
    'OFFSHORE WIND TURBINE' : ["WIND","WT"],
}

WECC = ["AZ","CA","CO","ID","MT","NM","OR","UT","WA","WY"]

class HIFLD:
    """HIFLD data accessor"""

    # pylint: disable=too-few-public-methods

    def __init__(self,
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        # pylint: disable=too-many-branches,too-many-locals
        drop_test:callable=None,
        drop_na:bool=True,
        drop_types:list[str]=None,
        drop_fuels:list[str]=None,
        geohash_precision:int=6,
        busdata:pd.DataFrame=None,
        groupby:list[str]=None,
        ):
        """Load HIFLD powerplant data

        
        drop_test: filter to remove unwanted records

        dropna: flag to enable dropping of records with NA data

        drop_types: list of plant type codes to exclude
        """
        self.powerplants = pd.read_csv("powerplants.csv.zip",
            usecols = [
                "NAME","COUNTY","STATE","TYPE","STATUS",
                "OPER_CAP","WINTER_CAP","SUMMER_CAP",
                "LATITUDE","LONGITUDE",
                ]
            )

        # filter plant list
        if drop_test:
            self.powerplants.drop(drop_test(self.powerplants),axis=0,inplace=True)

        # fix -999999 coding to be NaN
        for fixit in ["OPER_CAP","WINTER_CAP","SUMMER_CAP"]:
            self.powerplants[fixit] = [x if x > 0 else float('nan')
                for x in self.powerplants[fixit]]

        # drop nans if specified
        if drop_na:
            self.powerplants.dropna(inplace=True)

        # replace plant types with plant type codes
        self.powerplants["FUEL"] = ["|".join(sorted({PLANT_TYPES[y][0]
                for y in x.split("; ")})) for x in self.powerplants.TYPE]
        self.powerplants["TYPE"] = ["|".join(sorted({PLANT_TYPES[y][1]
                for y in x.split("; ")})) for x in self.powerplants.TYPE]

        # drop specified types
        self.powerplants["ID"] = self.powerplants.index
        self.powerplants.set_index("ID") # need "hard" index to drop inside iteration loop
        if drop_types:
            for n,row in self.powerplants.iterrows():
                for dtype in drop_types:
                    if dtype == row.TYPE: # plant is exclusive this type
                        self.powerplants.drop(n,inplace=True,axis=0)
                        break
                    types = row.TYPE.split("|")
                    if dtype in types: # plant has multiple types
                        types.remove(dtype)
                        ntype = "|".join(types)
                        row.TYPE = ntype
                        self.powerplants.loc[n,"TYPE"] = ntype

        # drop specified fuels
        if drop_fuels:
            for n,row in self.powerplants.iterrows():
                for dfuel in drop_fuels:
                    if dfuel == row.FUEL: # plant is exclusive this fuel
                        self.powerplants.drop(n,inplace=True,axis=0)
                        break
                    fuels = row.FUEL.split("|")
                    if dfuel in fuels: # plant has multiple fuels
                        fuels.remove(dfuel)
                        nfuel = "|".join(fuels)
                        row.TYPE = nfuel
                        self.powerplants.loc[n,"FUEL"] = nfuel

        # geohashing is enabled
        if not geohash_precision is None:

            # add plant geohashes
            self.powerplants["GEOCODE"] = [geohash(x,y,int(geohash_precision))
                for x,y in zip(self.powerplants.LATITUDE,self.powerplants.LONGITUDE)]

            # add bus geohash for plants
            if not busdata is None:
                buslist = {(x,y):g
                    for x,y,g in zip(busdata["LAT"],busdata["LON"],busdata["GEOHASH"])}
                nearest = list(zip(*(nearest2((x,y),buslist.keys())
                    for x,y in zip(self.powerplants["LATITUDE"],self.powerplants["LONGITUDE"]))))
                self.powerplants["BUSCODE"] = [geohash(x,y)
                    for x,y in nearest[1]]
                self.powerplants["BUSDIST"] = nearest[2]

        # no longer need "hard" index
        self.powerplants.reset_index(drop=True,inplace=True)

        # apply requested group and total capacities
        if groupby:

            self.powerplants = self.powerplants[groupby+["OPER_CAP","SUMMER_CAP","WINTER_CAP"]]\
                .groupby(groupby).sum().sort_index().reset_index()

        else:

            # sort by state, county, and name
            self.powerplants.sort_values(["STATE","COUNTY","NAME"],inplace=True)


if __name__ == "__main__":

    pd.options.display.width = None
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None

    # create list of allowed connection busses (e.g., only 20kV busses)
    gis = pd.read_csv("wecc240/gis.csv")
    bus = pd.read_csv("wecc240/bus.csv")
    busses = pd.merge(gis,bus,left_on="BUS_I",right_on="ID").drop("ID",axis=1)
    busses.drop(busses[busses.BASEKV>20.0].index,inplace=True)

    # get total load
    loads = pd.read_csv("wecc240/load.csv")[["PL","IP","YP"]].sum(axis=0).sum()

    # load HIFLD data
    print("All powerplant types")
    print("--------------------")
    data = HIFLD(
        drop_test=lambda x: x[(~x["STATE"].isin(WECC))|(x["STATUS"]!="OP")].index,
        # drop_types=["PV","WT","UNKNOWN"],
        # drop_fuels=["SUN","WIND"],
        busdata=busses[busses["BASEKV"]==20],
        groupby=["BUSCODE","TYPE"],
        )

    data.powerplants.set_index(["BUSCODE","TYPE"],inplace=True)
    print(f"{len(data.powerplants)} powerplants aggregated"
        " to {len(data.powerplants.index.get_level_values(0).unique())} 20kV busses")
    print("Plant types:",", ".join(set("|".join(["|".join(data.powerplants.index\
        .get_level_values(1).unique())]).split("|"))))
    capacity = data.powerplants.sum().round(1).to_dict()
    print(capacity)
    for name,value in capacity.items():
        print(f"{name} margin: {(1-loads/value)*100:.1f}%")

    print("\nNon-renewable/unknown plant types:")
    print("----------------------------------")
    data = HIFLD(
        drop_test=lambda x: x[(~x["STATE"].isin(WECC))|(x["STATUS"]!="OP")].index,
        drop_types=["PV","WT","UNKNOWN"],
        drop_fuels=["SUN","WIND"],
        busdata=busses[busses["BASEKV"]==20],
        groupby=["BUSCODE","TYPE"],
        )

    data.powerplants.set_index(["BUSCODE","TYPE"],inplace=True)
    print(f"{len(data.powerplants)} powerplants aggregated"
        " to {len(data.powerplants.index.get_level_values(0).unique())} 20kV busses")
    print("Plant types:",", ".join(set("|".join(["|".join(data.powerplants.index\
        .get_level_values(1).unique())]).split("|"))))
    capacity = data.powerplants.sum().round(1).to_dict()
    print(capacity)
    for name,value in capacity.items():
        print(f"{name} margin: {(1-loads/value)*100:.1f}%")

    print()
    print("Table 1 Comparison for HIFLD Powerplants")
    print("----------------------------------------")
    data = HIFLD(
        drop_test=lambda x: x[(~x["STATE"].isin(WECC))|(x["STATUS"]!="OP")].index,
        )
    data.powerplants.set_index(["STATE","FUEL"],inplace=True)
    data = data.powerplants["OPER_CAP"].groupby(["STATE","FUEL"]).sum().unstack("FUEL").fillna(0.0)
    mapping = { # map multiple to single
        'BIO|GAS': 'GAS', 
        'COAL|GAS': 'GAS', 
        'COAL|GAS|OIL': 'GAS', 
        'COAL|GAS|OTHER': 'GAS', 
        'COAL|SUN': 'SUN', 
        'COAL|GAS|OIL|WIND': 'WIND', 
        'ELEC|GAS': 'GAS', 
        'ELEC|OIL': 'PUMP', 
        'ELEC|WATER': 'PUMP', 
        'GAS|WOOD': 'GAS', 
        'GAS|OIL': 'GAS', 
        'GAS|OTHER': 'GAS', 
        'BIO|SUN': 'SUN', 
        'BIO|GAS|SUN': 'SUN', 
        'ELEC|SUN': 'SUN', 
        'ELEC|GAS|SUN': 'SUN', 
        'GAS|SUN': 'SUN', 
        'SUN|WIND': 'WIND', 
        'GAS|WASTE': 'GAS', 
        'BIO|WATER': 'WATER', 
        'OIL|WATER': 'WATER', 
        'WATER|WOOD': 'WATER',
        'WOOD' : 'BIO',
        'ELEC' : 'PUMP',
        'COKE' : 'COAL',
        'OIL' : 'COAL',
        'OTHER': 'GAS',
        'WASTE': 'BIO',
        }
    data["PUMP"] = 0.0
    data["DPV"] = 0.0
    for group,gtype in mapping.items():
        data[gtype] += data[group]
        data.drop(group,axis=1,inplace=True)
    reduced = ["BIO","COAL","GEO","GAS","WATER","NUC","ELEC","PUMP","SUN","WIND"]
    data["TOTAL"] = data.sum(axis=1)
    total = data.sum(axis=0)
    total.name = "TOTAL"
    data = pd.concat([data,pd.DataFrame(total).T])
    reorder = ["BIO","COAL","GEO","GAS","WATER","NUC","PUMP","SUN","WIND","DPV","TOTAL"]
    data.replace({0.0:""},inplace=True)
    print(data[reorder])