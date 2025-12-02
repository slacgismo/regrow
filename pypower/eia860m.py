"""EIA860m data"""

import os
import warnings
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
    "Entity ID": None,
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
FUELS = { # See https://www.eia.gov/survey/form/eia_860/instructions.pdf
    "BIT": "COAL",
    "LIG": "COAL",
    "PC": "COAL",
    "RC": "COAL",
    "SUB": "COAL",
    "WC": "COAL",
    "DFO": "OIL",
    "JF": "OIL",
    "LFG": "GAS",
    "NG": "GAS",
    "OBG": "GAS",
    "OG": "GAS",
    "PG": "GAS",
    "SUN": "SUN",
    "WND": "WIND",
    "WAT": "WATER",
    "NUC": "NUCLEAR",
    "AB": "WASTE",
    "BLQ": "WASTE",
    "WDL": "WASTE",
    "WDS": "WASTE",
    "MSW": "WASTE",
    "MWH": "OTHER",
    "OTH": "OTHER",
    "PUR": "OTHER",
    "WH": "OTHER",
    "GEO": "GEO",
}
GENS = { # See https://www.eia.gov/survey/form/eia_860/instructions.pdf
    "BA": "ES", # batteries
    "BT": "CC", # multi-cycle turbine (binary)
    "CA": "CC", # multi-cycle turbine (steam part)
    "CP": "ES", # concentrated solar storage
    "CS": "CC", # multi-cycle turbine (single shaft)
    "CT": "CC", # multi-cycle turbine (combustion part)
    "FC": "NA", # fuel cell
    "GT": "CT", # single-cycle turbine (combustion cycle)
    "HY": "HT", # hydro-electric turbine
    "IC": "IC", # internal combustion (diesel, etc.)
    "OT": "NA", # other (unknown)
    "PS": "ES", # pumped hydro storage
    "PV": "PV", # photo-voltaic
    "ST": "ST", # steam turbine
    "WT": "WT", # wind turbine
}

os.makedirs("wecc240/powerplants",exist_ok=True)

class EIA860(Generators):
    """EIA Form 860m generator data handler"""
    def __init__(self,
        year:int=2020,
        month:int=8,
        reload:bool=False,
        ):

        """Load generations from EIA Form 860 data

        Arguments:

        year: year of Form 860 data to load

        month: month of Form 860 data to load

        reload: flag to force reload from source rather than cache
        """

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

            # drop unwanted columns
            data.drop([x for x,y in MAPPING.items() if y is None ],axis=1,inplace=True)

            # convert to standard column names
            data.rename(MAPPING,axis=1,inplace=True)

            # add geohash values
            data["geohash"] = [geohash(x,y) for x,y in zip(data.latitude,data.longitude)]

            # index
            data = data[data["state"].isin(WECC)]\
                .set_index(["state","county","plant_id","generator_id","unit_code"])\
                .sort_index()

            # save to cache
            data.to_csv(file,index=True,header=True,compression="gzip")

        # load data from cache
        self.data = pd.read_csv(file,dtype=str)

        # initialize parent class
        super().__init__(source=url,cache=file)

if __name__ == "__main__":

    from wecc240 import wecc240
    eia860 = EIA860(reload=False)
    casedata = wecc240()

    pd.options.display.max_columns = None
    pd.options.display.width = None

    # test loading gen data into WECC 240 case
    gen = eia860.to_gen(
        case=casedata,
        converters={"fuel":FUELS,"gen":GENS},
        exclude={"fuel":["WIND","SUN","OTHER"]},
        # exclude={"fuel":["WIND","SUN","OTHER","OIL"],"gen":["IC","NA"]},
        )

    # test load gencost data into WECC240 case
    gencost = eia860.to_gencost(
        case=casedata,
        )

    casedata["gen"] = gen.values
    casedata["gencost"] = gencost.values

    from pypower.runpf import runpf
    from pypower.runopf import runopf
    from pypower.ppoption import ppoption

    result = {}

    pf,status = runpf(casedata,ppoption(VERBOSE=0,OUT_ALL=0))
    result["Powerflow time"] = f"{pf['et']*1000:.1f} ms" if status else 'FAILED'
    if status == 0:
        warnings.warn("EIA860m powerflow solution failed")

    opf = runopf(casedata,ppoption(VERBOSE=0,OUT_ALL=0))
    result["AC OPF stime"] = f"{opf['et']*1000:.1f} ms" if opf['success'] else 'FAILED'
    if opf['success'] == 0:
        warnings.warn("EIA860m AC OPF solution failed")
    opfpf,status = runpf(opf,ppoption(VERBOSE=0,OUT_ALL=0))
    if status == 0:
        warnings.warn("EIA860m AC OPF powerflow solution failed")
    result["AC OPF powerflow time"] = f"{opfpf['et']*1000:.1f} ms" if status else 'FAILED'

    gen.index.names = [
        "States",
        "Counties",
        "Nodes",
        "Busses",
        "Fuel types",
        "Generator types",
        ]
    for level in set(gen.index.names):
        gendata = gen.PMAX.groupby(level).sum().sort_values(ascending=False).to_frame()
        result[level] = len(gendata)
    result["Total generators"] = len(gen)
    result["Total capacity (GW)"] = round(float(gen.PMAX.sum()/1000),1)
    result["Operating cost ($M)"] = f"{opf["f"]*casedata["baseMVA"]/1000:.1f}"

    result = pd.DataFrame(result.values(),result.keys(),columns=["Result"])
    result.index.name = "EIA860m Summary"

    print(result)
