import os
import pandas as pd
import zipfile
import requests
from io import BytesIO

pd.options.display.max_columns = None
pd.options.display.width = None

STATES = ["AZ","CA","CO","ID","MT","NV","NM","OR","UT","WA","WY"]
URL = "https://www.eia.gov/electricity/data/eia861/archive/zip/{file}"

energy = []
for year in range(2018,2023):

    file = f"f861{year}.zip"
    url = URL.format(file=file)
    if not os.path.exists(file):
        print(f"Downloading {url}",end="...",flush=True)
        result = requests.get(url)
        assert result.status_code == 200, f"{file=} download failed"
        with open(file,"wb") as fh:
            fh.write(result.content)
        print("ok") 

    with zipfile.ZipFile(file) as fh:
        # print(fh.namelist())
        xlsx = BytesIO(fh.read(f"Sales_Ult_Cust_{year}.xlsx"))
        data = pd.read_excel(xlsx,
            skiprows=[0,1,2],
            usecols=[0,1,2,5,6,10,13,16,19,22],
            names=["year","id","name","dtype","state","residential[MWh]","commercial[MWh]","industrial[MWh]","transportation[MWh]","total[MWh]"],
            index_col=[0,4,1],

            converters={
                "residential[MWh]":lambda x:0.0 if x=="." else float(x),
                "commercial[MWh]":lambda x:0.0 if x=="." else float(x),
                "industrial[MWh]":lambda x:0.0 if x=="." else float(x),
                "transportation[MWh]":lambda x:0.0 if x=="." else float(x),
                "total[MWh]":lambda x:0.0 if x=="." else float(x),
            }
            ).dropna()
    # print(data.loc[[year],STATES,:])   
    data = pd.DataFrame(data.loc[[year],STATES,:][["residential[MWh]","commercial[MWh]","industrial[MWh]","transportation[MWh]","total[MWh]"]].groupby("state").sum())
    data["year"] = year
    energy.append(data.reset_index().set_index(["year","state"]))

pd.concat(energy).to_csv("eia_f861.csv")