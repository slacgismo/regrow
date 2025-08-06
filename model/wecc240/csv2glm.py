import os
import sys
import datetime as dt
import pandas as pd
import numpy as np

pd.options.display.width = None

usecols = {
    "branch" : ["from","to","status","rateA"],
}

zone = {
    1: "AL",
    2: "CA",
    3: "AZ",
    4: "BC",
    5: "CO",
    6: "ID",
    7: "MX",
    8: "MT",
    9: "NV",
    10: "NM",
    11: "OR",
    12: "UT",
    13: "WA",
    14: "WY",
    }

def read_bus():
    bus = pd.read_csv("bus.csv",
        usecols=["id","zone","Gs","Bs","Vmin","Vmax"],
        index_col="id",
        converters={
            "zone": int,
            "id" : int,
            "Gs": float,
            "Bs": float,
            "Vmin": float,
            "Vmax": float,
            },
        )
    gis = pd.read_csv("gis.csv",
        usecols=["id","name","baseKV","type","latitude","longitude"],
        index_col="id",
        converters={
            "id" : int,
            "baseKV" : float,
            "type" : int,
            "latitude" : float,
            "longitude" : float,
            },
        )
    bus = bus.join(gis,on="id")
    bus.rename({"zone":"area"},axis=1,inplace=True)
    bus = fix_names(bus)
    return bus

def read_gen():
    gen = pd.read_csv("gen.csv",
        usecols=["parent","name","Pg","Qg","Qmax","Qmin","Vg","status","Pmax","Pmin"],
        index_col="name",
        )
    return gen

def read_branch():
    branch = pd.read_csv("branch.csv",
        usecols=["from","to","status","rate2"],
        converters={
            "from": int,
            "to": int,
            "status": int,
            "rateA" : float,
            "r": float,
            "x": float,
            },
        )
    branch["name"] = [f"L_{x['from']:.0f}_{x['to']:.0f}" for _,x in branch.iterrows()]
    branch.rename({"rate2":"rateA"},axis=1,inplace=True)
    return branch.set_index("name")

def read_transformer():
    xformer = pd.read_csv("transformer.csv",
        usecols=["from","to","rateA","r","x"],
        converters={
            "from": int,
            "to": int,
            "rateA": float,
            "r": float,
            "x": float,
        }
        )
    xformer["name"] = [f"T_{x['from']:.0f}_{x['to']:.0f}" for _,x in xformer.iterrows()]
    xformer["ratio"] = 1.0
    return xformer.set_index("name")

def read_shunt():
    shunt = pd.read_csv("shunt.csv",
        usecols=["parent","control_mode","status","voltage_high","voltage_low",
            "remote_bus","admittance","steps_1","admittance_1",
            ],
        converters={
            "parent": int,
            "control_mode": int,
            "status": int,
            "voltage_high": float,
            "admittance": float,
            "step_1": int,
            "admittance_1": float,
        }
        )
    shunt["name"] = [f"SL_{x['parent']:.0f}" for n,x in shunt.iterrows()]
    shunt["parent"] = [f"bus:{x['parent']:.0f}" for _,x in shunt.iterrows()]
    shunt["remote_bus"] = [f"bus:{x['remote_bus']:.0f}" for _,x in shunt.iterrows()]
    shunt["dwell_time"] = 3600.0
    shunt["control_gain"] = 0.0
    shunt = fix_names(shunt)
    return shunt.set_index("name")

def read_load():
    load = pd.read_csv("load.csv")
    load["name"] = [f"LD_{x['parent']:.0f}_{n}" for n,x in load.iterrows()]
    load["parent"] = [f"bus:{x['parent']:.0f}" for _,x in load.iterrows()]
    load["Z"] = [f"{x['Pz']:.1f}{x['Qz']:+.1f}j" for _,x in load.iterrows()]
    load["I"] = [f"{x['Pi']:.1f}{x['Qi']:+.1f}j" for _,x in load.iterrows()]
    load["P"] = [f"{x['Pp']:.1f}{x['Qp']:+.1f}j" for _,x in load.iterrows()]
    load.drop(["Pp","Qp","Pi","Qi","Pz","Qz"],axis=1,inplace=True)
    load = fix_names(load)
    return load.set_index("name")

def fix_names(data):
    # finds and fixes duplicate names
    found = {x:0 for x in data["name"].unique()}
    for n,x in data.iterrows():
        data.loc[n,"name"] += f"_{found[x['name']]}"
        found[x['name']] += 1
    return data


def fix_branches(branch,bus):

    # interpolate missing rateA values
    branch["fromKV"] = [bus.loc[x].baseKV for x in branch["from"]]
    branch["toKV"] = [bus.loc[x].baseKV for x in branch["to"]]
    if "r" in branch.columns:
        branch["g"] = 1.0 / branch["r"]
    if "x" in branch.columns:
        branch["b"] = 1.0 / branch['x']
    nz = branch[branch.rateA>0]
    # nz[["fromKV","rateA"]].plot(kind="scatter",x="fromKV",y="rateA").figure.savefig("branch.png")
    fit = np.poly1d(np.polyfit(nz.fromKV,nz.rateA,1))
    KVlevels = branch[["fromKV","rateA"]].groupby("fromKV")
    branch.rateA = [x.rateA if x.rateA > 0 else fit(x.fromKV) for n,x in branch.iterrows()]
    branch = pd.DataFrame(
        data={
            "rateA": branch["rateA"].groupby("name").sum().round().tolist(),
            "r": (1/branch["g"].groupby("name").sum()).round(6).tolist() if "g" in branch.columns else 0,
            "x": (1/branch["b"].groupby("name").sum()).round(6).tolist() if "b" in branch.columns else 0,
        },
        index=branch["rateA"].groupby("name").count().index
        )
    branch[["from","to"]] = [[int(y) for y in x.split('_')[1:]] for x in branch.index]
    branch["fromKV"] = [bus.loc[x].baseKV for x in branch["from"]]
    branch["toKV"] = [bus.loc[x].baseKV for x in branch["to"]]

    # fix from/to references
    branch["from"] = [f"bus:{x['from']:.0f}" for _,x in branch.iterrows()]
    branch["to"] = [f"bus:{x['to']:.0f}" for _,x in branch.iterrows()]

    # set status
    branch['status'] = 1
    branch['ratio'] = 1

    # from unneeded data
    branch.drop(["fromKV","toKV"],inplace=True,axis=1)

    return branch

def write_glm(file,oclass,data):
    with open(file,"w") as fh:
        print("module pypower;",file=fh)
        for n,values in data.reset_index().iterrows():
            if "id" in values:
                print(f"object pypower.{oclass}:{values['id']}",file=fh)
            else:
                print(f"object pypower.{oclass}",file=fh)
            print("{",file=fh)
            for tag,value in [(x,y) for x,y in values.items() if x not in {"id","index"}]:
                print(f"    {tag} {value};",file=fh)
            print("}",file=fh)
    with open("model.glm","a") as fh:
        print(f'#include "{file}"',file=fh)
    print(f"{oclass=}:",len(data),"objects")

def main(*args,**kwargs):

    #
    # bus
    #
    bus = read_bus()

    with open("model.glm","w") as fh:
        print(f"// created by {' '.join(sys.argv)}",file=fh)
        print(f"// created at {dt.datetime.now()}",file=fh)
        print(f"""module pypower
{{
    autosize_angle 80 deg;
}}""",file=fh)
    write_glm("bus.glm","bus",bus)

    #
    # gen
    #
    gen = read_gen()
    write_glm("gen.glm","gen",gen)

    #
    # branch
    #
    branch = fix_branches(read_branch(),bus)
    branch.drop(["r","x","ratio"],axis=1,inplace=True)
    write_glm("branch.glm","branch",branch)

    #
    # transformers
    #
    transformer = fix_branches(read_transformer(),bus)
    write_glm("transformer.glm","branch",transformer)

    #
    # shunts
    #
    shunt = read_shunt()
    write_glm("shunt.glm","shunt",shunt)

    #
    # loads
    #
    load = read_load()
    write_glm("load.glm","load",load)

if __name__ == "__main__":

    main(
        [x for x in sys.argv if "=" not in x],
        dict([(x.split("=",1)) for x in sys.argv if "=" in x])
        )
