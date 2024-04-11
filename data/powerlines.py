"""Convert HIFLD powerlines database to GLM pypower data"""

import os, sys
import datetime as dt
import pandas as pd

pd.options.display.max_columns = None
pd.options.display.width = None

csvfile = sys.argv[1] if len(sys.argv)>1 else "powerlines.csv.zip"
powerlines = pd.read_csv(csvfile, 
    index_col=["ID"],
    usecols = ["ID","TYPE","STATUS","VOLTAGE","SUB_1","SUB_2","SHAPE__Length"],
    converters = {
        "TYPE" : lambda x: "|".join([y.replace(" ","_") for y in x.split("; ")]),
        "STATUS" : lambda x: x=="IN SERVICE",
        "SUB_1" : lambda x: x.replace(" ","_"),
        "SUB_2" : lambda x: x.replace(" ","_"),
        "SHAPE__Length" : lambda x: round(float(x),3),
        }
    )


glmfile = sys.argv[2] if len(sys.argv)>2 else csvfile.replace(".csv.zip",".glm")

sublist = {}
n_subs = 0

with open(glmfile,"w") as fh:
    print(f"""// generated from '{" ".join(sys.argv)}' on {dt.datetime.now()}

module pypower;

""",file=fh)

    for ID,data in powerlines.iterrows():
        lname = f"L_{ID}";
        xfmr = False
        for sub in [data.SUB_1,data.SUB_2]:
            nname = f"N_{n_subs}"
            if not sub in sublist:
                sublist[sub] = {
                    "name" : nname,
                    "voltage" : data.VOLTAGE,
                    "links" : [lname],
                    }
                n_subs += 1
            elif sublist[sub]["voltage"] != data.VOLTAGE:
                if sublist[sub]["voltage"] < 0 and data.VOLTAGE > 0:
                    sublist[sub]["voltage"] = data.VOLTAGE
                elif data.VOLTAGE > 0:
                    if data.SHAPE__Length > 100:
                        print(f"""WARNING: voltage mismatch at bus {nname} for branches {lname} (V={data.VOLTAGE} kV) and L_{sublist[sub]["links"][0]} (V={sublist[sub]["voltage"]} kV)""")
                    else:
                        xfmr = True
            sublist[sub]["links"].append(lname)
        print(f"""object pypower.branch
{{
    name "L_{ID}";
    from "{sublist[data.SUB_1]['name']}";
    to "{sublist[data.SUB_2]['name']}";
}}
""",file=fh)
        if xfmr:
            print(f"""object pypower.transformer
{{
    parent "L_{ID}";
    // type {data.TYPE};
    status {"IN" if data.STATUS else "OUT"};
}};""",file=fh)
        else:
            print(f"""object pypower.powerline
{{
    parent "L_{ID}";
    // type {data.TYPE};
    status {"IN" if data.STATUS else "OUT"};
    length {data.SHAPE__Length} ft;
}};""",file=fh)

    for ID,data in sublist.items():
        print(f"""object pypower.bus
{{
    name "{ID}";
    baseKV {data["voltage"]} kV;
    // links {",".join(data['links'])}
}}""",file=fh)
