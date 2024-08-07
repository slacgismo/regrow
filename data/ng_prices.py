"""Update natural gas prices

Syntax: python3 -m ng_prices OPTION [...]

Options
-------
    -h|--help|help  generate this help document
    --inputs        generate list of input files
    --outputs       generate list of output files
    --startdate=YYYY-MM-YY change start date (default is 2018-01-01)
    --stopdate=YYYY-MM-YY change stop date (default is 2022-01-01)
    --update        update only missing/outdates files
    --verbose       verbose progress updates

Description
-----------

Downloads the latest daily natural gas prices at the Henry hub.
"""
import pandas as pd
import datetime as dt

INPUTS={
	"EIADATA":"https://www.eia.gov/dnav/ng/xls/NG_PRI_FUT_S1_D.xls",
	}
OUTPUTS={
	"PRICES":"ng_prices.csv",
	}
INTERNAL=[]

STARTDATE="2018-01-01"
STOPDATE="2022-01-01"

from utils import *

options.context = "sensitivity.py"

for arg in read_args(sys.argv,__doc__):
	if arg == "--inputs":
		print(" ".join(INPUTS.values()))
		exit(0)
	elif arg == "--outputs":
		print(" ".join(OUTPUTS.values()))
		exit(0)
	elif arg.startswith("--startdate"):
		STARTDATE=arg.split("=")[1]
	elif arg.startswith("--startdate"):
		STOPDATE=arg.split("=")[1]
	elif arg != "--update":
		raise Exception(f"option '{arg}' is not valid")

STARTDATE=pd.to_datetime(STARTDATE,format="%Y-%m-%d",utc=True)
STOPDATE=pd.to_datetime(STOPDATE,format="%Y-%m-%d",utc=True)

if __name__ == "__main__":
	data = pd.read_excel(INPUTS["EIADATA"],
		sheet_name="Data 1",
		header=0,
		skiprows=2,
		names=["datetime","price[$/MBtu]"],
		index_col="datetime",
		)
	data.index = pd.to_datetime(data.index,format="%Y-%m-%d",utc=True)
	data = data.resample(rule="1D").ffill()
	data.drop(data.iloc[(data.index<STARTDATE-dt.timedelta(hours=1)) | (data.index>STOPDATE)].index,inplace=True)
	# print(data.dropna().resample(rule="1D").ffill())
	data.dropna().to_csv(OUTPUTS["PRICES"],index=True,header=True)
