"""Load model testing"""

import marimo as mo
import os
import sys
import datetime as dt
import pandas as pd
import numpy as np
import re
from tzinfo import TIMEZONES, TZ
import states

NAME = os.path.splitext(os.path.split(__file__)[1])[0]

TZINFO={
    "EST" : TZ("EST",-5,0),
    "CST" : TZ("CST",-6,0),
    "MST" : TZ("MST",-7,0),
    "PST" : TZ("PST",-8,0),
}
CACHEDIR="./geodata/buildings"
VERBOSE = True

def verbose(msg):
    print(f"VERBOSE [{NAME}]: {msg}",file=sys.stderr)


class County:
    """County data"""
    _counties = pd.read_csv("counties.csv",index_col="geocode")
    _regions = {
        "WECC" : ["AZ","CA","CO","ID","MT","NM","NV","OR","UT","WA","WY"]
    }

    def __init__(self,geocode:str):
        """Load county data"""
        self.geocode = geocode
        data = self._counties.loc[geocode].to_dict()
        data["state"] = data["usps"]
        data["fips"] = f"{data['fips']:05.0f}"
        data["name"] = f"g{data['fips'][:2]}0{data['fips'][2:]}0"
        data["cache"] = f"{CACHEDIR}/{data['name']}"
        os.makedirs( data["cache"],exist_ok=True)
        for name,value in data.items():
            setattr(self,name,value)
        try:
            self.timezone = TZINFO[TIMEZONES[self.fips][:3]]
        except:
            self.timezone = TZINFO[TIMEZONES[self.fips[:2]][:3]]

    def __str__(self):
        return f"{self.county} {self.state} ({self.geocode})"

    def __repr__(self):
        return f"{NAME}.{__class__.__name__}({repr(self.geocode)})"

    def to_dict(self) -> dict:
        """Convert county data to dict"""
        return {x:getattr(self,x) for x in dir(self) if not callable(getattr(self,x)) and not x.startswith("_")}

    @classmethod
    def states(self,pattern:str=".*") -> dict:
        """Get states matching pattern or region name"""
        if pattern in self._regions:
            return self._regions[pattern]
        return {x:states.state_codes_byusps[x] for x in self._counties.usps.unique() if re.match(pattern,x)}

    @classmethod
    def counties(self,states:list[str],asdict=False) -> list|dict:
        """Get counties in state(s)"""
        if isinstance(states,str):
            states = [states]
        if asdict:
            data = {}
            counties = self._counties.reset_index().set_index(["usps","geocode"])
            for state in states:
                data[state] = list(counties.loc[state].index.get_level_values(0).values)
            return data
        return list(self._counties[self._counties.usps.isin(states)].index)

class Weather:
    """Weather data"""
    _server = "https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2021/resstock_amy2018_release_1/weather/amy2018/{fips}_2018.csv"

    def __init__(self,county:County):
        """Load weather data"""
        name = f"G{county.fips[:2]}0{county.fips[2:]}0"
        file = f"{county.cache}/weather.csv"
        if not os.path.exists(file):
            url = self._server.format(fips=county.name)
            try:
                data = pd.read_csv(url,
                    index_col = [0],
                    usecols = [0,1,3,5],
                    parse_dates = [0],
                    dtype = float,
                    low_memory = True,
                    header=None,
                    skiprows=1,
                    )
                data.columns = ["temperature[degC]","wind[m/s]","solar[W/m^2]"]
                data.index = data.index.tz_localize(county.timezone).tz_convert("UTC").tz_localize(None)-dt.timedelta(hours=1) # localize and change to leading timestamp
                data.index.name = "timestamp"
                self.data = data.resample("1h").mean().round(1)
                self.data.to_csv(file,header=True,index=True)
            except Exception as err:
                print(f"ERROR [{repr(url)}]: {err}",file=sys.stderr)
                raise
        else:
            self.data = pd.read_csv(file,index_col=[0],parse_dates=[0],low_memory=True)
        self.index = np.array([int((float(x)-float(self.data.index.values[0]))/3600e9) for x in self.data.index.values])
        self.timestamp = self.data.index.tz_localize("UTC").tz_convert(county.timezone)
        self.units = {}
        for column in [x for x in self.data.columns if "[" in x]:
            cname,cunit = column.split("[")
            setattr(self,cname,np.array(self.data[column].values))
            self.units[cname] = cunit.strip("]")
        self.county = county

    def __str__(self):
        return f"{self.county} weather"

    def __repr__(self):
        return f"{NAME}.{__class__.__name__}({repr(self.county.geocode)})"

    def __getitem__(self,name:str):
        return self.data[name]

class Load:
    """Load data"""

    _loads = ["total","baseline","heating","cooling"]
    _buildings = {
        "residential" : {
            "single-family_detached" : "House",
            "single-family_attached" : "Townhouse",
            "multi-family_with_2_-_4_units" : "Small apartment/condo",
            "multi-family_with_5plus_units" : "Large apartment/condo",
            "mobile_home" : "Mobile home",
        },
        "commercial" : {
            "largeoffice": "Large office",
            "secondaryschool" : "Large school",
            "largehotel" : "Large hotel",
            "hospital" : "Hospital",
            "mediumoffice" : "Medium Office",
            "retailstripmall" : "Medium retail",
            "outpatient" : "Healthcare",
            "smalloffice" : "Small office",
            "retailstandalone" : "Small retail",
            "primaryschool" : "Small school",
            "smallhotel" : "Small hotel",
            "fullservicerestaurant" : "Restaurant",
            "quickservicerestaurant" : "Fast food",
            "warehouse" : "Warehouse",
        }
    }
    _sectors = list(_buildings.keys())
    _servers = {
        "residential": "https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2021/resstock_amy2018_release_1/timeseries_aggregates/by_county/state={usps}/{fips}-{building}.csv",
        "commercial": "https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2021/comstock_amy2018_release_1/timeseries_aggregates/by_county/state={usps}/{fips}-{building}.csv",
    }
    _usecols = {
        "residential": [
            "timestamp",
            "out.electricity.cooling.energy_consumption",
            "out.electricity.heating.energy_consumption",
            "out.electricity.heating_supplement.energy_consumption",
            "out.electricity.total.energy_consumption",
        ],
        "commercial": [
            "timestamp",
            "out.electricity.cooling.energy_consumption",
            "out.electricity.heating.energy_consumption",
            "out.electricity.heating_supplement.energy_consumption",
            "out.electricity.total.energy_consumption",
        ]
    }

    def __init__(self,county:County,
        sectors:list[str]=None,
        loads:list[str]=None,
        buildings:list[str]=None,
        ):
        """Get load data"""
        
        assert isinstance(county,County), f"{county=} is not a County object"
        self.county = county

        self.sectors = self._sectors if sectors is None else sectors
        for sector in self.sectors:
            assert sector in self._sectors, f"{repr(sectors)} is not a valid sector"
        
        self.loads = self._loads if loads is None else loads
        for load in self.loads:
            assert load in self._loads, f"{repr(loads)} is not a valid load"

        def concat(x):
            result = []
            for y in x:
                result.extend(y)
            return result

        _buildings = concat([list(self._buildings[x].keys()) for x in self.sectors])
        self.buildings = _buildings if buildings is None else buildings
        for building in _buildings:
            assert building in _buildings, f"{repr(buildings)} is not a valid building"

        self.data = None
        for building in self.buildings:
            sector = self.sector(building)
            name = f"G{county.fips[:2]}0{county.fips[2:]}0"
            file = f"{county.cache}/{building}.csv"
            if not os.path.exists(file):
                verbose(f"Downloading {building}...")
                url = self._servers[sector].format(usps=county.usps,fips=county.name,building=building)
                try:
                    data = pd.read_csv(url,
                        index_col=["timestamp"],
                        usecols=self._usecols[sector],
                        parse_dates=["timestamp"],
                        converters={
                            "out.electricity.cooling.energy_consumption": lambda x: float(
                                x
                            )
                            / 1000,
                            "out.electricity.heating.energy_consumption": lambda x: float(
                                x
                            )
                            / 1000,
                            "out.electricity.heating_supplement.energy_consumption": lambda x: float(
                                x
                            )
                            / 1000,
                            "out.electricity.total.energy_consumption": lambda x: float(
                                x
                            )
                            / 1000,
                        },
                        low_memory=True,
                        )
                    data.columns = [
                        "cooling[MW]",
                        "heating[MW]",
                        "auxheat[MW]",
                        "total[MW]",
                    ]
                    data["heating[MW]"] += data["auxheat[MW]"]
                    data.drop("auxheat[MW]", axis=1, inplace=True)
                    data.index = (
                        data.index.tz_localize("EST" if sector == "commercial" else county.timezone)
                        .tz_convert("UTC")
                        .tz_localize(None)
                    )
                    data.index = data.index - dt.timedelta(
                        minutes=15
                    )  # change lagging timestamps to leading timestamp
                    data = pd.DataFrame(data.resample("1h").sum()).round(6)
                    data.to_csv(file, header=True, index=True)
                except Exception as err:
                    print(f"ERROR [{repr(url)}]: {err}",file=sys.stderr)
                    data = None
            else:
                verbose(f"Reloading {building}...")
                data = pd.read_csv(file,index_col=[0],parse_dates=[0],low_memory=True)

            if not data is None:
                if self.data is None:
                    self.data = data
                else:
                    self.data = self.data + data

        if not self.data is None:
            self.index = np.array([int((float(x)-float(self.data.index.values[0]))/3600e9) for x in self.data.index.values])
            self.timestamp = self.self.data.index.tz_localize("UTC").tz_convert(county.timezone)
            self.units = {}
            for column in [x for x in self.data.columns if "[" in x]:
                cname,cunit = column.split("[")
                setattr(self,cname,np.array(self.data[column].values))
                self.units[cname] = cunit.strip("]")
        self.county = county

    def __str__(self):
        return f"{self.county} {'/'.join(self.sectors)} loads"

    def __repr__(self):
        return f"Load.{self.county}(county={repr(self.county)},sectors={repr(self.sectors)},loads={repr(self.loads)},buildings={repr(self.buildings)})"

    @classmethod
    def building(self,names:list|str) -> str|dict:
        """Get building name(s)"""
        if isinstance(names,list):
            result = {}
            for name in names:
                result[name] = self.building(name)
            return result
        for sector in self._sectors:
            if names in self._buildings[sector]:
                return self._buildings[sector][names]
        raise ValueError(f"{repr(names)} is not a valid building type")

    def sector(self,building:str) -> str:
        """Get building's sector"""
        for sector,buildings in self._buildings.items():
            if building in buildings:
                return sector
        raise ValueError(f"{repr(building)} is not a valid building type")

if __name__ == "__main__":

    # test global county data access
    # print(County.states("C"))
    # print(County.counties(County.states("WECC")))
    # print(County.counties(County.states("WECC"),asdict=True))

    # test local county data access
    county = County("9q9q1v")
    # print(county)
    # print(repr(county))
    # print(county.to_dict())

    # test weather data access
    weather = Weather(county)
    # print(Weather(county))
    # print(repr(Weather(county)))
    # print(weather.units)
    # print(weather.timestamp)
    # print(weather.temperature)

    # test load data access
    # print(Load.building("largeoffice"))
    # print(Load.building(["largeoffice","smalloffice"]))
    load = Load(county)
    # print(load)
    # print(repr(load))
