"""Create weather geodata files

Syntax: python3 -m weather OPTIONS [...]

Options
-------

    -h|--help|help Generate this help

    --inputs       Generate list of input files

    --outputs      Generate list of output files

    --update       Update weather folder

    --verbose      Enable verbose output (default False)

    --year=YEAR    Set the year (default 2020)

    --freq=FREQ    Set the sampling frequency (default 1h)

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

YEAR = 2020
FREQ = "1h"
ROUND = 1

if __name__ == "__main__":

    if len(sys.argv) == 1:
        print("".join([x for x in __doc__.split("\n") if x.startswith("Syntax: ")]))
    
    if "--inputs" in sys.argv:
        print(' '.join(INPUTS.values()))

    if "--outputs" in sys.argv:
        print(' '.join(OUTPUTS.values()))

    if "--verbose" in sys.argv:
        options.verbose = True

    for arg in sys.argv[1:]:
        if arg.startswith("--year="):
            YEAR = int(arg.split("=")[1])
        elif arg.startswith("--freq="):
            FREQ = arg.split("=")[1]
        elif arg.startswith("--round="):
            ROUND = int(arg.split("=")[1])
        if arg in ["-h","--help","help"]:
            print(__doc__)

    if "--update" in sys.argv:

        # load WECC bus data
        gis = pd.read_csv("wecc240_gis.csv",index_col=['Bus  Number'])
        gis["geocode"] = [geohash(x,y,6) for x,y in gis[["Lat","Long"]].values]
        results = {}
        for location in sorted(gis["geocode"].unique()):
            verbose(f"Processing {location}",end="...")
            data = pd.DataFrame(nsrdb_weather(location,YEAR).resample(FREQ).mean())
            # print(data,file=sys.stderr)
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
