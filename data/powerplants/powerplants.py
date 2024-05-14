import os, sys
import pandas as pd
import math

pd.options.display.max_columns = None
pd.options.display.width = None

def toascii(text,replace=""):
    return "".join([x if 32<=ord(x)<=127 else replace for x in text])

def tounit(value):
    if type(value) is float:
        return "" if math.isnan(value) else f"_{int(value)}"
    else:
        return f"_{value}"

for file in os.listdir("."):
    if file.endswith(".csv") and not file.endswith("_cost.csv"):
        plants = pd.read_csv(file)

        with open(os.path.splitext(file)[0]+".glm","w") as glm:
            print("module pypower;",file=glm)
            for _,data in plants.iterrows():
                try:
                    print(f"""object powerplant
{{
    name "{toascii(data['name'].upper().replace(" ","_"))}{tounit(data['unit'])}";
    // type "{data['type']}";
    // county "{data['county']}";
    state "{data['state']}";
    latitude {data['latitude']};
    longitude {data['longitude']};
    out_svc {'INIT' if data['start[y]'] < 1980 else '"'+str(int(data['start[y]']))+'-01-01 00:00:00 UTC"'};
    out_svc {'NEVER' if math.isnan(data['retirement[y]']) else '"'+str(int(data['retirement[y]']))+'-01-01 00:00:00 UTC"'};
    operating_capacity "{data['capacity[MW]']} MW";
}}
""",file=glm)
                except Exception as err:
                    print(f"EXCEPTION [powerplants.py]: {err}",file=sys.stderr)
                    print("DATA RECORD:",data,file=sys.stderr)
