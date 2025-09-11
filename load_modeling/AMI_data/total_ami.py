import os
import pandas as pd

pd.options.display.max_columns = None
pd.options.display.width = None

CACHE = "data.csv"

data = pd.read_csv(CACHE)

dates = zip(data["record_date"],data["hour_id"],data["utc_offset"])
index = [f"{ymd} {h-1:02d}:00:00{-tz:+02d}:00" for ymd,h,tz in dates]

data.index =pd.DatetimeIndex(pd.to_datetime(index,utc=True))
data.index.name = "datetime[utc]"
data.drop(["record_date","hour_id","utc_offset"],inplace=True,axis=1)
data = (data/1000).round(3)
data.columns=["load[MW]"]

print(data,flush=True)

