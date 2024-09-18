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

YEARS = "2018,2019,2020,2021,2022"
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
            panel = {}
            for name in [x.split("[")[0] for x in OUTPUTS]:
                panel[name] = []
            verbose("ok")

        else:

            panel = {}
            verbose("Loading old weather data",end="...")
            for channel,file in OUTPUTS.items():
                name = channel.split("[")[0]
                panel[name] = pd.read_csv(file,
                    index_col=[0],
                    parse_dates=[0],
                    date_format="%Y-%m-%d %H:%M:%S+00:00",
                    )
                panel[name].index = panel[name].index.tz_localize("UTC")
            verbose("ok")

        # load WECC bus data
        nodes = pd.read_csv(INPUTS["NETWORK"],index_col=['Bus  Number'])
        nodes["geocode"] = [geohash(x,y,6) for x,y in nodes[["Lat","Long"]].values]
        for location in sorted(nodes["geocode"].unique()):
            verbose(f"Processing {location}",end="...")
            for year in [int(x) for x in YEARS.split(',')]:
                data = None
                for channel in channel_names:
                    name = channel.split("[")[0]
                    if len(panel[name][location][panel[name][location].index.year==year].dropna()) == 0:
                        if data is None:
                            data = pd.DataFrame(nsrdb_weather(location,year,30,channel_names).resample(FREQ).mean()).round(ROUND)
                        if not location in panel[name].columns:
                            panel[name].loc[pd.to_datetime(data[channel].index.values,utc=True),location] = data[channel].values.astype('float')
                        elif int(year) not in panel[name].index.year.unique():
                            panel[name] = pd.concat([
                                panel[name].astype(float),
                                pd.DataFrame(
                                    data[channel].values,
                                    pd.to_datetime(data[channel].index.values,utc=True),
                                    columns=[location]).astype(float),
                                ])
                        else:
                            panel[name].loc[data[channel].index,location] = data[channel].values
            for channel,file in OUTPUTS.items():
                name = channel.split('[')[0]
                try:
                    panel[name].to_csv(file,index=True,header=True)
                except KeyboardInterrupt:
                    panel[name].to_csv(file,index=True,header=True)
                    raise
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
