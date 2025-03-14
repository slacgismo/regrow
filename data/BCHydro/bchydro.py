"""Load BC Hydro tieline data and output generation and load data for BC Hydro"""
import os
import datetime as dt
import pandas as pd
import xlrd as xl

# load data from BC Hydro
data = []
for year in range(2018,2023):
    url = f"https://www.bchydro.com/content/dam/BCHydro/customer-portal/documents/corporate/suppliers/transmission-system/actual_flow_data/historical_data/HourlyTielineData{year}.xls"
    data.append(pd.read_excel(url,skiprows=1,header=[0],parse_dates=["Date"]))
data = pd.concat(data)

# set date index
data["Date"] = pd.DatetimeIndex(data.Date) + pd.to_timedelta(data.HE,"h") - pd.to_timedelta([1]*len(data),"h")
data = data.set_index("Date").sort_index()

# compute tieline "generation"
data["generation"] = [max(x,0) for x in data["US Tielines"]]
data["generation"].to_csv("bchydro_gen.csv",header=True,index=True)

# compute tieline "load"
data["load"] = [max(-x,0) for x in data["US Tielines"]]
data["load"].to_csv("bchydro_load.csv",header=True,index=True)
