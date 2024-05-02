"""Create weather geodata files

Syntax: python3 -m weather OPTIONS [...]

Options
-------

    -h|--help|help    Generate this help

    --inputs          Generate list of input files

    --outputs         Generate list of output files

    --update          Update weather folder

    --verbose         Enable verbose output (default False)

    --years=YEAR,...  Set the year(s) (default 2020)

    --freq=FREQ       Set the sampling frequency (default 1h)

Description
-----------

Generate the weather folder CSV files. Weather files are formatted as tables
with locations in columns and timestamps in rows, e.g.,

timestamp,GEOCODE_1,GEOCODE_2,...,GEOCODE_N
TIMESTAMP_1,VALUE_1_1,VALUE_1_2,...,VALUE_1_N
TIMESTAMP_2,VALUE_2_1,VALUE_2_2,...,VALUE_2_N
...
TIMESTAMP_T,VALUE_T_1,VALUE_T_2,...,VALUE_T_N


Credentials
-----------

Your NSRDB credentials must be stored in ~/.nsrdb/credentials.json in the format

    {"email" : "apikey"}

"""

import os, sys
import json
import pandas as pd

from utils import *
options.context = "weather.py"
options.verbose = False

INPUTS = {}

OUTPUTS = {
    "solar[W/m^2]" : "weather/solar.csv",
    "temperature[degC]" : "weather/temperature.csv",
    "wind[m/s]" : "weather/wind.csv",
}

YEARS = "2018,2019,2020,2021"
FREQ = "1h"
ROUND = 1

if __name__ == "__main__":

    if len(sys.argv) == 1:
        print("".join([x for x in __doc__.split("\n") if x.startswith("Syntax: ")]))
        exit(E_MISSING)
    
    elif "-h" in sys.argv or "--help" in sys.argv or "help" in sys.argv:
        print(__doc__)
        exit(E_OK)

    elif "--inputs" in sys.argv:
        print(' '.join(INPUTS.values()))
        exit(E_OK)

    elif "--outputs" in sys.argv:
        print(' '.join(OUTPUTS.values()))
        exit(E_OK)

    elif "--update" in sys.argv:
        if "--verbose" in sys.argv:
            options.verbose = True
        for arg in sys.argv[1:]:
            if arg.startswith("--years="):
                YEARS = arg.split("=")[1]
            elif arg.startswith("--freq="):
                FREQ = arg.split("=")[1]
            elif arg.startswith("--round="):
                ROUND = int(arg.split("=")[1])
            elif not arg in ["--verbose","--update"]:
                error(E_INVAL,f"option '{arg}' is invalid")

        # load WECC bus data
        gis = pd.read_csv("wecc240_gis.csv",index_col=['Bus  Number'])
        gis["geocode"] = [geohash(x,y,6) for x,y in gis[["Lat","Long"]].values]
        results = {}
        for location in sorted(gis["geocode"].unique()):
            verbose(f"Processing {location}",end="...")
            data = []
            for year in [int(x) for x in YEARS.split(',')]:
                verbose(".",end="")
                data.append(pd.DataFrame(nsrdb_weather(location,year).resample(FREQ).mean()))
            data = pd.concat(data)
            for column in data.columns:
                if not column in results:
                    results[column] = []
                results[column].append(pd.DataFrame(data={location:data[column]},index=data.index).round(ROUND))
            verbose("ok")

        verbose("Creating weather folder")
        os.makedirs("weather",exist_ok=True)

        for name,data in results.items():
            verbose(f"Saving {OUTPUTS[name]}",end="...")
            pd.concat(data,axis=1).to_csv(OUTPUTS[name],header=True,index=True)
            verbose("ok")
