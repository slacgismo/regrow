"""Create WECC model"""

import pandas as pd

bus = pd.read_csv("wecc_bus.csv",index_col=[0],comment="#").to_dict('index')
branch = pd.read_csv("wecc_branch.csv",comment="#").to_dict('index')

with open("wecc.glm","w") as fh:
    print("module pypower;",file=fh)

    for name,data in branch.items():

        fbus,tbus = data["from"],data["to"]

        if fbus in bus:
            fbus = bus[fbus]["name"]
        else:
            bus[fbus] = {"name":fbus}

        if tbus in bus:
            tbus = bus[tbus]["name"]
        else:
            bus[tbus] = {"name":tbus}

        assert fbus != tbus, f"branch {name} from {fbus} same as to {tbus}"
        properties = "\n    ".join([f"{x} {y}" for x,y in data.items() if x in ["voltage[kV]","path"] and str(float(y)) != 'nan'])
        print(f"""object branch
{{
    name "{data['from']}_{data['to']}";
    from "{bus[data['from']]['name']}";
    to "{bus[data['to']]['name']}";
    {properties}
}}
""",file=fh)

    for name,data in bus.items():
        latlon = "\n    ".join([f"{x} {float(y)};" for x,y in data.items() if x in ["latitude","longitude"] and str(float(y)) != 'nan'])
        print(f"""object bus
{{
    name "{data['name']}";
    {latlon}
}}""",file=fh)
