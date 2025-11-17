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

Example:
    data = HIFLD(
        drop_test=lambda x: x[(~x["STATE"].isin(WECC))|(x["STATUS"]!="OP")].index,
        drop_types=["PV","WT"],
        drop_fuels=["SUN","WIND"],
        )
    print(data.powerplants)
"""

import pandas as pd
from geohash import geohash

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
        # pylint: disable=too-many-branches
        drop_test:callable=None,
        drop_na:bool=True,
        drop_types:list[str]=None,
        drop_fuels:list[str]=None,
        geohash_precision:int=6,
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
        self.powerplants["FUEL"] = ["|".join({PLANT_TYPES[y][0]
                for y in x.split("; ")}) for x in self.powerplants.TYPE]
        self.powerplants["TYPE"] = ["|".join({PLANT_TYPES[y][1]
                for y in x.split("; ")}) for x in self.powerplants.TYPE]

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

        # add plant geohashes
        if not geohash_precision is None:
            self.powerplants["GEOCODE"] = [geohash(x,y,int(geohash_precision))
                for x,y in zip(self.powerplants.LATITUDE,self.powerplants.LONGITUDE)]

        # sort by state, county, and name
        self.powerplants.sort_values(["STATE","COUNTY","NAME"],inplace=True)

        # no longer need "hard" index
        self.powerplants.reset_index(drop=True,inplace=True)

if __name__ == "__main__":

    pd.options.display.width = None
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None

    data = HIFLD(
        drop_test=lambda x: x[(~x["STATE"].isin(WECC))|(x["STATUS"]!="OP")].index,
        drop_types=["PV","WT","UNKNOWN"],
        drop_fuels=["SUN","WIND"],
        )
    print(len(data.powerplants),"powerplants selected for WECC")
    print("types:",", ".join(set("|".join(["|".join(data.powerplants.TYPE.unique())]).split("|"))))
    print("fuels:",", ".join(set("|".join(["|".join(data.powerplants.FUEL.unique())]).split("|"))))
