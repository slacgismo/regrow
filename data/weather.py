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
options.debug = True

INPUTS = {
    "NETWORK" : "wecc240_gis.csv",
}

OUTPUTS = {
    "solar[W/m^2]" : "weather/solar.csv",
    "temperature[degC]" : "weather/temperature.csv",
    "wind[m/s]" : "weather/wind.csv",
}

channel_names = {
    "solar[W/m^2]" : "ghi",
    "temperature[degC]" : "temp_air",
    "wind[m/s]" : "wind_speed",
}

YEARS = "2018,2019,2020,2021"
FREQ = "1h"
ROUND = 1

def main(YEARS=YEARS,FREQ=FREQ,ROUND=ROUND):

    global INPUTS
    global OUTPUTS
    global channel_names
    if len(sys.argv) == 1:
        print("".join([x for x in __doc__.split("\n") if x.startswith("Syntax: ")]))
    
    elif "-h" in sys.argv or "--help" in sys.argv or "help" in sys.argv:
        print(__doc__)

    elif "--inputs" in sys.argv:
        print(' '.join(INPUTS.values()))

    elif "--outputs" in sys.argv:
        print(' '.join(OUTPUTS.values()))

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

        if not os.path.exists("weather"):
            
            verbose("Creating weather folder",end="...")
            os.makedirs("weather",exist_ok=True)
            verbose("ok")

        else:

            results = {}
            verbose("Loading old weather data",end="...")
            for file in os.listdir("weather"):
                data = pd.read_csv(os.path.join("weather",file),
                    index_col=[0],
                    parse_dates=[0],
                    date_format="%Y-%m-%d %H:%M:%S+00:00",
                    )
                data.index = data.index.tz_localize("UTC")
                name = os.path.splitext(file)[0]
                results[name] = {}
                for year in data.index.year.unique():
                    results[name][year] = {}
                    for location in data.columns:
                        result = data[location][data.index.year==year].dropna()
                        if len(result) > 0:
                            results[name][year][location] = pd.DataFrame(result.values,result.index,columns=[location])
            verbose("ok")

        # load WECC bus data
        gis = pd.read_csv(INPUTS["NETWORK"],index_col=['Bus  Number'])
        gis["geocode"] = [geohash(x,y,6) for x,y in gis[["Lat","Long"]].values]
        for location in sorted(gis["geocode"].unique()):
            verbose(f"Processing {location}",end="...")
            for year in [int(x) for x in YEARS.split(',')]:
                # verbose(".",end="")
                data = None
                for channel in OUTPUTS:
                    channel_name = channel.split('[')[0]
                    if channel_name not in results:
                        results[channel_name] = {}
                    if year not in results[channel_name]:
                        results[channel_name][year] = {}
                    if location not in results[channel_name][year]:
                        data = pd.DataFrame(nsrdb_weather(location,year,30,channel_names).resample(FREQ).mean()).round(ROUND)
                        verbose(".",end="")    
                        for name,column in channel_names.items():
                            cname = name.split("[")[0]
                            if year not in results[cname]:
                                results[cname][year] = {}
                            # verbose(f"({cname},{year},{location})")
                            results[cname][year][location] = pd.DataFrame(data[name].values,data.index,columns=[location])
            verbose(".",end="")    
            for channel,file in OUTPUTS.items():
                name = channel.split('[')[0]
                data = []
                for year,years in sorted(results[name].items()):
                    result = pd.DataFrame(pd.concat(years.values(),axis=1),columns=[location])
                    data.append(result)
                results[name][location] = pd.concat(data,axis=0)
                print(results[name])
                quit()
                pd.concat(list(results[name].values()),axis=1).to_csv(file,index=True,header=True)
            verbose("ok")   

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        error(E_INTR,"keyboard interrupt")
    except Exception as err:
        if options.debug:
            raise
        error(E_INTR,err)