import os, sys
import pandas as pd

result = []
for file in os.listdir("."):
    if not file.endswith(".csv.zip"):
        continue
    state,county = file.split('.')[0].split('-')
    data = pd.read_csv(file,index_col = ["State","BuildingType"]
                      )
    FloorArea = data.groupby(["State","BuildingType"])["FloorArea"].sum()
    WindowWallRatio = (data["WindowWallRatio"]*data["FloorArea"]).groupby(["State","BuildingType"]).sum()/FloorArea
    Height = (data["Height"]*data["FloorArea"]).groupby(["State","BuildingType"]).sum()/FloorArea
    GroundArea = (data["GroundArea"]*data["FloorArea"]).groupby(["State","BuildingType"]).sum()/FloorArea
    latitude = (data["latitude"]*data["FloorArea"]).groupby(["State","BuildingType"]).sum()/FloorArea
    longitude = (data["longitude"]*data["FloorArea"]).groupby(["State","BuildingType"]).sum()/FloorArea
    _data = {
        "latitude" : latitude.round(6),
        "longitude" : longitude.round(6),
        "state" : state,
        "county" : county.replace('_',' '),
        "floorarea[Msf]" : (FloorArea/1e6).round(3),
        "WindowWallRatio[pu]" : WindowWallRatio.round(2),  
        "Height[ft]" : Height.round(1),
        "GroundArea[ksf]" : (GroundArea/1e3).round(3),
    }
    result.append(pd.DataFrame(_data))
result = pd.concat(result)
result.to_csv("buildings.csv",index=True,header=True)
