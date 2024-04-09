"""Assemble commercial building inventory by county"""

import pandas as pd
pd.options.display.max_columns = None
pd.options.display.width = None

result = []
for region in ["west"]:
    data = pd.read_csv(f"commercial_buildings_{region}_2019.csv.zip",low_memory=False)
    data = data.loc[data["statecode"].isin(['AZ','CA','CO','ID','MT','NM','NV','OR','UT','WA','WY'])]
    data["countyname"] = [x.replace(" County","") for x in data["countyname"]]
    data.set_index(["statecode","countyname","doe_prototype"],inplace=True)
    result.append(pd.DataFrame(data.groupby(["statecode","countyname","doe_prototype"])["area_sum"].sum()))

pd.concat(result).to_csv("commercial_floorarea.csv",index=True,header=True)
