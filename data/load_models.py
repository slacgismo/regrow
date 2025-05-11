"""Load models from NREL ResStock/ComStock data

~~~
>>> import load_models
~~~

Here are some frequently used data/methods:

* List of counties

~~~
>>> load_models.County._counties
        usps   fips               county   latitude  longitude
geocode                                                       
djf3h6    AL   1001       Autauga County  32.532237 -86.646440
dj3w7m    AL   1003       Baldwin County  30.659218 -87.746067
djem29    AL   1005       Barbour County  31.870253 -85.405104
djf5c6    AL   1007          Bibb County  33.015893 -87.127148
dn43q1    AL   1009        Blount County  33.977358 -86.566440
...      ...    ...                  ...        ...        ...
de2bcp    PR  72145  Vega Baja Municipio  18.455128 -66.397883
de1rp5    PR  72147    Vieques Municipio  18.125418 -65.432474
de0xpk    PR  72149   Villalba Municipio  18.130718 -66.472244
de1ntr    PR  72151    Yabucoa Municipio  18.059858 -65.859871
de0qys    PR  72153      Yauco Municipio  18.085669 -66.857901

[3220 rows x 5 columns]
~~~

* List of states

~~~
>>> load_models.County._counties.usps.unique()
array(['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA',
       'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA',
       'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY',
       'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
       'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR'], dtype=object)
~~~

* List of states in a region

~~~
>>> load_models.County._regions["WECC"]
['AZ', 'CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']
~~~

* County geocodes for a state:

~~~{y.county:x for x,y in load_models.County._counties.iterrows() if y.usps
== "NV"}
{'Churchill County': '9r5buv', 'Clark County': '9qqnnb', 'Douglas
County': '9qfvq4', 'Elko County': '9rmfp1', 'Esmeralda
County': '9qsqgj', 'Eureka County': '9rj76j', 'Humboldt
County': '9r7gxf', 'Lander County': '9rhfbx', 'Lincoln
County': '9qwq45', 'Lyon County': '9qgjzx', 'Mineral County': '9qgg17', 'Nye
County': '9qtpvz', 'Pershing County': '9r5y19', 'Storey
County': '9r5025', 'Washoe County': '9r4zsy', 'White Pine
County': '9rn21r', 'Carson City': '9qfyep'}
~~~


"""

import marimo as mo
import os
import sys
import datetime as dt
import pandas as pd
import numpy as np
import re
from tzinfo import TIMEZONES, TZ
import states
import matplotlib.pyplot as plt
import pwlf

NAME = os.path.splitext(os.path.split(__file__)[1])[0]

TZINFO={
    "EST" : TZ("EST",-5,0),
    "CST" : TZ("CST",-6,0),
    "MST" : TZ("MST",-7,0),
    "PST" : TZ("PST",-8,0),
}
CACHEDIR="./geodata/buildings"
VERBOSE = False # enable verbose output
WARNING = True # enable warning output
DEBUG = False # enable traceback on exception handlers

def verbose(msg):
    if VERBOSE:
        print(f"VERBOSE [{NAME}]: {msg}",file=sys.stderr)

def warning(msg):
    if WARNING:
        print(f"WARNING [{NAME}]: {msg}",file=sys.stderr)

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
            verbose(f"Downloading {county} weather...")
            url = self._server.format(fips=name)
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
            verbose(f"Reloading {county} weather...")
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
        return f"<{self.county} weather>"

    def __repr__(self):
        return f"{NAME}.{__class__.__name__}({repr(self.county.geocode)})"

    def __getitem__(self,name:str):
        return self.data[name]

class Loads:
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
    _columns = {
            "timestamp": "timestamp",
            "out.electricity.cooling.energy_consumption": "cooling[MW]",
            "out.electricity.heating.energy_consumption": "heating[MW]",
            "out.electricity.heating_supplement.energy_consumption": "auxheat[MW]",
            "out.electricity.total.energy_consumption": "total[MW]",
        }
    _converters = {
        "out.electricity.cooling.energy_consumption": lambda x: float(x) / 1000,
        "out.electricity.heating.energy_consumption": lambda x: float(x) / 1000,
        "out.electricity.heating_supplement.energy_consumption": lambda x: float(x) / 1000,
        "out.electricity.total.energy_consumption": lambda x: float(x) / 1000,
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
                verbose(f"Downloading {county} {building}...")
                url = self._servers[sector].format(usps=county.usps,fips=county.name,building=building)
                try:
                    data = pd.read_csv(url,
                        index_col=["timestamp"],
                        usecols=lambda x: x in self._columns,
                        parse_dates=["timestamp"],
                        converters=self._converters,
                        low_memory=True,
                        )
                    for column in self._converters:
                        if column not in data.columns:
                            warning(f"{repr(column)} not found in {repr(building)} data for {repr(name)}")
                            data[column] = 0.0
                    data.rename(self._columns,axis=1,inplace=True)
                    data["heating[MW]"] += data["auxheat[MW]"]
                    data["baseload[MW]"] = data["total[MW]"] - data["cooling[MW]"] - data["heating[MW]"]
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
                    warning(f"no {repr(building)} data ({url=}, {err=})")
                    data = None
                    if DEBUG:
                        raise
            else:
                verbose(f"Reloading {county} {building}...")
                data = pd.read_csv(file,index_col=[0],parse_dates=[0],low_memory=True)

            if not data is None:
                self.data = data if self.data is None else (self.data + data)

        if self.data is None:
            raise Exception(f"{county} {building} has no data")

        self.index = np.array([int((float(x)-float(self.data.index.values[0]))/3600e9) for x in self.data.index.values])
        self.timestamp = self.data.index.tz_localize("UTC").tz_convert(county.timezone)
        self.units = {}
        for column in [x for x in self.data.columns if "[" in x]:
            cname,cunit = column.split("[")
            setattr(self,cname,np.array(self.data[column].values))
            self.units[cname] = cunit.strip("]")

        self.county = county

    def __str__(self):
        return f"<{self.county} {'/'.join(self.sectors)} {'/'.join(self.buildings)} loads>"

    def __repr__(self):
        return f"Load.{self.county}(county={repr(self.county)},sectors={repr(self.sectors)},loads={repr(self.loads)},buildings={repr(self.buildings)})"

    @classmethod
    def building(self,buildings:list|str) -> str|dict:
        """Get building name(s)

        Arguments:
        
        * `buildings`: building type name(s)

        Returns:

        * `str`: readable name of building type

        * `dict`: readable names of building types 
        """
        if isinstance(buildings,list):
            result = {}
            for building in buildings:
                result[building] = self.building(building)
            return result
        for sector in self._sectors:
            if buildings in self._buildings[sector]:
                return self._buildings[sector][buildings]
        raise ValueError(f"{repr(buildings)} is not a valid building type")

    @classmethod
    def sector(self,building:str) -> str:
        """Get building's sector

        Arguments:

        * `building`: get sector of a building type

        Returns:
        """
        for sector,buildings in self._buildings.items():
            if building in buildings:
                return sector
        raise ValueError(f"{repr(building)} is not a valid building type")

class Model:
    """Load model

    Parameters:

    * `growth`: building type load growth rates

    * `eletrification`: fossil end-use electrification rates

    * `upgrades`: electric end-use efficiency improvement rates
    """
    def __init__(self,*args):
        """Create a model from county, weather, and/or load data

        Arguments:
    
        * `county`: county object
    
        * `weather`: weather object
    
        * `loads`: load object
        """
        self.args = args
        self.county = None
        self.weather = None
        self.loads = None
        self.holdout = []
        self.growth = {} # annual load growth
        self.electrification = {} # end-use electrification rates
        self.upgrades = {} # end-use technology upgrades

        for data in args:

            if isinstance(data,County):

                self.county = data

            elif isinstance(data,Weather):

                self.weather = data

            elif isinstance(data,Loads):

                self.loads = data

            else:

                raise TypeError(f"{data=} is not a County, Weather, or Loads object")

        if self.county is None and self.weather is None and self.loads is None:

            raise ValueError("you must provide at least a County, Weather, or Load object")

        if self.county is None:
            self.county = self.weather.county if self.weather else self.loads.county
        if self.weather is None:
            self.weather = Weather(self.county)
        if self.loads is None:
            self.loads = Loads(self.county)

        self.results = {
            "residuals": None,
            "RMSE": None,
            "HeatingModel": None,
            "CoolingModel": None,
            "BaseloadModel": None,
        }

    def __str__(self):

        return f"<{self.county} {__class__.__name__}>"

    def __repr__(self):

        return f"{NAME}.{__class__.__name__}({','.join([repr(x) for x in self.args])})"

    def fit(self):
        raise TypeError(f"cannot fit using the abstract class of Model")

    def predict(self):
        raise TypeError(f"cannot predict using the abstract class of Model")

class NERCModel(Model):
    """Implement the simple NERC load forecasting model"""
    def fit(self,cutoff:float=0):
        """Fit model

        Arguments:

        * `cutoff`: heating/cooling load cutoff value below which load is ignored
        """
        totals = self.loads.data


        # heating and cooling fit
        for load in ["heating","cooling"]:

            data = self.weather.data.join(totals[totals[f"{load}[MW]"]>cutoff]).dropna()
            X = data["temperature[degC]"].values
            Y = data[f"{load}[MW]"].values
            weights = 1/Y.std()

            self.results[f"{load.title()}Model"] = pwlf.PiecewiseLinFit(X,Y,weights=weights)
            self.results[f"{load.title()}Model"].fit(2)

        # baseload fit
        data = self.weather.data.join(totals).dropna()
        Y = data["baseload[MW]"].values
        X = data["temperature[degC]"].values
        weights = 1/Y.std()
        test = np.arange(min(X),max(X))
        self.results["BaseloadModel"] = pwlf.PiecewiseLinFit(X,Y,weights=weights)
        self.results["BaseloadModel"].fit(1)

        # holdout test (use all training data if no holdout)
        Y = data["total[MW]"].values
        if self.holdout:
            X,Y = X[self.holdout],Y[self.holdout]
        baseload = self.results["BaseloadModel"].predict(X)
        heating = self.results["HeatingModel"].predict(X)
        cooling = self.results["CoolingModel"].predict(X)
        self.results["residuals"] = Y - baseload - heating - cooling
        rmse = np.sqrt(np.linalg.norm(self.results["residuals"]))
        self.results["RMSE [MW]"] = f"""{rmse:.1f} MW"""
        self.results["RMSE [%]"] = f"""{rmse/data["total[MW]"].mean()*100:.1f} %"""
        self.results["Heating temperature"] = f"""{self.results["HeatingModel"].fit_breaks[1]:.1f} degC"""
        self.results["Heating sensitivity"] = f"""{self.results["HeatingModel"].beta[1]:.1f} MW/degC"""
        self.results["Cooling temperature"] = f"""{self.results["CoolingModel"].fit_breaks[1]:.1f} degC"""
        self.results["Cooling sensitivity"] = f"""{self.results["CoolingModel"].beta[1]:.1f} MW/degC"""

    def predict(self,
        temperatures:list[float],
        loads:list[str]=["baseload","cooling","heating"],
        ) -> list[float]:
        """Predict loads

        Arguments:

        * `temperatures`: temperatures at which to predict load

        * `loads`: load to include in prediction (must be among models in results)

        Returns:

        * `list[float]`: predictions for temperatures given
        """
        return sum([self.results[f"{x.title()}Model"].predict(temperatures) for x in loads])

if __name__ == "__main__":

    # test global county data access
    assert list(County.states("C").keys()) == ["CA","CO","CT"], "County.states(str) failed"
    assert County.counties(County.states("WECC"))[0] == "9w61k3", "County.counties(str) failed"
    assert County.counties(County.states("WECC"),asdict=True)["AZ"][0] == "9w61k3", "County.counties(str,asdict) failed"

    # test local county data access
    county = County("9q9q1v")
    assert str(county) == "Alameda County CA (9q9q1v)", "County.__str__() failed"
    assert repr(county) == "load_models.County('9q9q1v')", "County.__repr__() failed"
    # print(county.to_dict())

    # test weather data access
    weather = Weather(county)
    assert str(Weather(county)) == "<Alameda County CA (9q9q1v) weather>", "Weather.__str__()"
    assert repr(Weather(county)) == "load_models.Weather('9q9q1v')", "Weather.__repr__() failed"
    assert weather.units["temperature"] == "degC", "weather.units() failed"
    assert str(weather.timestamp[0]) == "2018-01-01 00:00:00-08:00", "weather.timestamp failed"
    assert weather.temperature[0] == 8.9, "weather.temperature failed"

    # test load data access
    assert Loads.building("largeoffice") == "Large office", "Loads.building(str) failed"
    assert Loads.building(["largeoffice","smalloffice"]) == {'largeoffice': 'Large office', 'smalloffice': 'Small office'}, "Loads.building(list[str]) failed"
    assert Loads.sector("largeoffice") == "commercial", "Loads.sector(str) failed"
    load = Loads(county)
    assert str(load) == "<Alameda County CA (9q9q1v) residential/commercial single-family_detached/single-family_attached/multi-family_with_2_-_4_units/multi-family_with_5plus_units/mobile_home/largeoffice/secondaryschool/largehotel/hospital/mediumoffice/retailstripmall/outpatient/smalloffice/retailstandalone/primaryschool/smallhotel/fullservicerestaurant/quickservicerestaurant/warehouse loads>", "Loads.__str__() failed"
    assert repr(load) == "Load.Alameda County CA (9q9q1v)(county=load_models.County('9q9q1v'),sectors=['residential', 'commercial'],loads=['total', 'baseline', 'heating', 'cooling'],buildings=['single-family_detached', 'single-family_attached', 'multi-family_with_2_-_4_units', 'multi-family_with_5plus_units', 'mobile_home', 'largeoffice', 'secondaryschool', 'largehotel', 'hospital', 'mediumoffice', 'retailstripmall', 'outpatient', 'smalloffice', 'retailstandalone', 'primaryschool', 'smallhotel', 'fullservicerestaurant', 'quickservicerestaurant', 'warehouse'])", "Loads.__repr__() failed"
    load.data.plot(figsize=(30,10),title=load.county)
    # plt.show()

    model = Model(county)
    model = Model(weather)
    model = Model(load)
    model = NERCModel(load)
    model.fit()
    assert round(model.predict(np.arange(-20,40))[0],1) == 867.0, "Model.predict() failed"
