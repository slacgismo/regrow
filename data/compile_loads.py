import marimo

__generated_with = "0.1.82"
app = marimo.App(width="full")


@app.cell
def __(pd):
    #
    # Load bus gis data
    #
    gis = pd.read_csv("wecc240_gis.csv")
    return gis,


@app.cell
def __(os, pd):
    #
    # Process buildings in county files
    #
    if os.path.exists("building_data.csv"):
        data = pd.read_csv("building_data.csv",index_cols=["County","BuildingType"])
    else:
        data = []
        for _file in [x for x in os.listdir("buildings") if x.endswith(".csv.zip")]:
            _county = _file.split(".")[0]
            _data = pd.read_csv(
                os.path.join("buildings", _file),
                usecols=["FloorArea", "BuildingType", "latitude", "longitude"],
            )
            _building_type = _data.groupby("BuildingType")
            _result = pd.DataFrame(_building_type.sum()["FloorArea"]/1e6).round(2)
            _result["BuildingCount"] = _building_type.count()["FloorArea"]
            _result["County"] = _county
            data.append(_result.reset_index())
        data = pd.concat(data)
        data.set_index(["County","BuildingType"],inplace=True)
        data.to_csv("building_data.csv",index=True,header=True)
    data
    return data,


@app.cell(disabled=True, hide_code=True)
def __(array, gen, gis):
    #
    # Match buildings to busses based on gis data
    #
    _xx,_yy = array(gis.Lat),array(gis.Long)
    names = gis[gis.columns[0]]
    with open("powerplants_gis.glm","w") as _fh:
        for _name,_plant in gen.items():
            if not "latitude" in _plant:
                print(f"// powerplant {_name} does not have a location",file=_fh)
                continue
            _x,_y = float(_plant["latitude"]),float(_plant["longitude"])
            _dx,_dy = _xx-_x,_yy-_y
            _d = zip(_dx*_dx + _dy*_dy,names)
            _dn = sorted(_d,key=lambda x:x[0])
            print("modify",f"{_name}.parent",f"wecc240_psse_N_{_dn[0][1]};",file=_fh)
    return names,


@app.cell
def __():
    import os, sys
    import marimo as mo
    import pandas as pd
    pd.options.display.max_rows=100
    return mo, os, pd, sys


if __name__ == "__main__":
    app.run()
