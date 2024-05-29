import os, sys
sys.path.insert(0,"..")
import pandas as pd
import math
import utils

pd.options.display.max_columns = None
pd.options.display.width = None

fuels = {
    # dispatchable
    "oil.csv" : "OIL",
    "gas.csv" : "NG",
    "geothermal.csv" : "GEO",
    "bioenergy.csv" : "BIO",
    "coal.csv" : "COAL",
    # non-dispatchable (flat or played)
    "nuclear.csv" : "NUC",
    "hydro.csv" : "WATER",
    # intermittent
    "wind.csv" : "WIND",
    "solar.csv" : "SUN",
}

dispatchable = ["OIL","NG","GEO","BIO","COAL"]

def toascii(text,replace=""):
    if type(text) is float and math.isnan(text):
        return ""
    else:
        return "".join([x if 32<=ord(x)<=127 else replace for x in text])

def tounit(value):
    if type(value) is float:
        return "" if math.isnan(value) else f"_{int(value)}"
    else:
        return f"_{value.replace(' ','_')}"

def toname(value):
    name = toascii(data['name'].upper().replace(" ","_").replace(".",""))
    return ("_"+name) if '0' <= name[0] <= '9' else name

def totimestamp(value,nan='INVALID',init=1980,never=3000):
    if type(value) is float and math.isnan(value):
        return nan
    elif int(value) < init:
        return 'INIT'
    elif int(value >= never):
        return 'NEVER'
    else:
        return f'"{int(value)}-01-01 00:00:00 UTC"'

weccgis = pd.read_csv("../wecc240_gis.csv",usecols=["Bus  Number","Lat","Long"])
weccgis.columns=["bus_id","latitude","longitude"]
weccgis["geocode"] = [utils.geohash(x,y,6) for x,y in weccgis[["latitude","longitude"]].values]
weccgis.set_index("geocode",inplace=True)
weccgis = weccgis[~weccgis.index.duplicated(keep='first')]

for file in os.listdir("."):
    if file.endswith(".csv") and not file.endswith("_cost.csv"):

        plants = pd.read_csv(file)
        if os.path.exists(file.replace(".csv","_cost.csv")):
            costdata = pd.read_csv(file.replace(".csv","_cost.csv"))
            print(costdata)
        else:
            costdata = None

        with open(os.path.splitext(file)[0]+".glm","w") as glm:

            print(f"""module pypower;
class powerplant
{{
    char32 county;
}}
""",file=glm)
            for _,data in plants.iterrows():
                nearest = utils.nearest(utils.geohash(data['latitude'],data['longitude'],6),weccgis.index)
                busid = int(weccgis.loc[nearest]["bus_id"])
                name = toname(data['name']) + tounit(data['unit'])
                hasgen = fuels[file] in dispatchable
                if hasgen:
                    print(f"""object gen
{{
    name "wecc240_psse_G_{toname(data['name'])}{tounit(data['unit'])}";
    parent "wecc240_psse_N_{busid}";
}}""",file=glm)
                try:
                    print(f"""object powerplant
{{
    name "{name}";
    parent "wecc240_psse_{"G" if hasgen else "N"}_{name if hasgen else busid}";
    generator "{data['type']}";
    fuel "{fuels[file]}";
    county "{toascii(data['county'])}";
    state "{data['state']}";
    latitude {data['latitude']};
    longitude {data['longitude']};
    in_svc {totimestamp(data['start[y]'],nan='INIT')};
    out_svc {totimestamp(data['retirement[y]'],nan='NEVER')};
    operating_capacity "{data['capacity[MW]']} MW";
}}
""",file=glm)
                except Exception as err:
                    print(f"EXCEPTION [powerplants.py]: {err}",file=sys.stderr)
                    print("DATA RECORD:",data,file=sys.stderr)
