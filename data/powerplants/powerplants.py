import os, sys
import pandas as pd
import math

pd.options.display.max_columns = None
pd.options.display.width = None

fuels = {
    "nuclear.csv" : "NUC",
    "hydro.csv" : "WATER",
    "oil.csv" : "OIL",
    "gas.csv" : "NG",
    "wind.csv" : "WIND",
    "solar.csv" : "SUN",
    "geothermal.csv" : "GEO",
    "bioenergy.csv" : "BIO",
    "coal.csv" : "COAL",
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
    name = toascii(data['name'].upper().replace(" ","_"))
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

for file in os.listdir("."):
    if file.endswith(".csv") and not file.endswith("_cost.csv"):
        plants = pd.read_csv(file)

        with open(os.path.splitext(file)[0]+".glm","w") as glm:
            number = "TODO"
            print(f"""module pypower;
class powerplant
{{
    char32 county;
}}
""",file=glm)
            for _,data in plants.iterrows():
                try:
                    print(f"""object powerplant
{{
    name "{toname(data['name'])}{tounit(data['unit'])}";
    parent "pp_{"gen" if fuels[file] in dispatchable else "bus"}_{number}";
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
