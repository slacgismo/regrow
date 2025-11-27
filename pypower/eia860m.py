import os
import datetime as dt
import calendar as cal
import pandas as pd
from geohash import geohash
from ppgen import Generators

WECC = ["AZ","CA","CO","ID","MT","NM","OR","UT","WA","WY"]
FILE = "wecc240/powerplants/eia860m_{date}.csv.gz"
URL = "https://www.eia.gov/electricity/data/eia860m/archive/xls/{month}_generator{year}.xlsx"
MAPPING = {
    "Plant State": "state",
    "County": "county",
    "Plant ID": "plant_id",
    "Generator ID": "generator_id",
    "Unit Code": "unit_code",
    "Entity ID": "owner_id",
    "Plant Name": "plant_name",
    "Nameplate Capacity (MW)": "operating_capacity",
    "Net Summer Capacity (MW)": "summer_capacity",
    "Net Winter Capacity (MW)": "winter_capacity",
    "Technology": "technology",
    "Energy Source Code": "fuel",
    "Prime Mover Code": "gen",
    "Latitude": "latitude",
    "Longitude": "longitude",
    }

os.makedirs("wecc240/powerplants",exist_ok=True)

class EIA860(Generators):

    def __init__(self,
        year:int=2020,
        month:int=8,
        reload:bool=False
        ):

        # convert date to EIA URL filename format
        self.date = dt.date(year,month,1)
        file = FILE.format(date=self.date) 
        month = cal.month_name[month].lower()
        url = URL.format(year=year,month=month)

        # get data if not in cache
        if not os.path.exists(file) or reload:
            data = pd.read_excel(url,
                sheet_name="Operating",
                skiprows=[0,22987],
                usecols=[0,2,3,5,6,7,8,9,10,11,12,13,25,26,27,],
                )
            data.rename(MAPPING,axis=1,inplace=True)
            data["geohash"] = [geohash(x,y) for x,y in zip(data.latitude,data.longitude)]
            data = data[data["state"].isin(WECC)]\
                .set_index(["state","county","plant_id","generator_id","unit_code"])\
                .sort_index()

            data.to_csv(file,index=True,header=True,compression="gzip")

        # load data
        self.data = pd.read_csv(file,dtype=str)

        # initialize parent class 
        super().__init__(source=url,cache=file)

if __name__ == "__main__":

    fleet = EIA860(reload=False)
    pd.options.display.max_columns = None
    pd.options.display.width = None
    print(fleet.data)
    from wecc240 import wecc240
    print(fleet.to_gen(case=wecc240()))

