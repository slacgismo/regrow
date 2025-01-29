import os
import sys
import datetime as dt

loads = open("loads.csv","w")
print("datetime,location,status,voltage[pu*V],load[MVA]",file=loads) # ,impedance,current,power,voltage,nominal,response

locations = {}
accessors = {}
def on_commit(t0):
    for obj in [x for x in gridlabd.get("objects") if x.startswith("wecc240_psse_L_")]:
        if obj not in accessors:
            # data = [data[x] for x in ['status','S','Z','I','P','V','Vn','response']]
            accessors[obj] = {
                "power" : gridlabd.property(obj,"S"),
                "voltage" : gridlabd.property(obj,"V"),
                "nominal" : gridlabd.property(obj,"Vn"),
                "status" : gridlabd.property(obj,"status")
                }
        data = accessors[obj]
        S = abs(data["power"].get_value())
        V = abs(data["voltage"].get_value()) / data["nominal"].get_value()
        status = data["status"].get_value()

        if obj not in locations:
            data = gridlabd.get_object(obj)
            parent = gridlabd.get_object(data["parent"])
            lat,lon = parent["latitude"],parent["longitude"]
            geo = gridlabd.get_global(f"GEOCODE {lat},{lon}#6")
            locations[obj] = geo
        else:
            geo = locations[obj]
        print(f"{dt.datetime.fromtimestamp(t0)},{geo},{status},{V:.3f},{S:.1f}",file=loads)
    return True
