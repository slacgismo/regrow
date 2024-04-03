import marimo

__generated_with = "0.3.4"
app = marimo.App(width="full")


@app.cell
def __(ns, pd):
    #
    # Load bus gis data
    #
    bus = pd.read_csv("wecc240_gis.csv")
    bus["geocode"] = [ns.geohash(x,y,6) for x,y in zip(bus["Lat"],bus["Long"])]
    bus.set_index("geocode",inplace=True)
    bus.sort_index(inplace=True)

    return bus,


@app.cell
def __(pd):
    #
    # Load building data
    #
    buildings = pd.read_csv("buildings.csv").set_index(["geocode","BuildingType"]).sort_index()
    return buildings,


@app.cell
def __(array, buildings, bus):
    _gis = bus.reset_index()
    _xx,_yy = array(_gis.Lat),array(_gis.Long)
    _names = _gis[_gis.columns[0]]
    with open("buildings_gis.glm","w") as _fh:
        for _name,_data in buildings.iterrows():
            if not "latitude" in _data:
                print(f"// building {'_'.join(_name)} does not have a location",file=_fh)
                continue
            _x,_y = float(_data["latitude"]),float(_data["longitude"])
            _dx,_dy = _xx-_x,_yy-_y
            _d = zip(_dx*_dx + _dy*_dy,_names)
            _dn = sorted(_d,key=lambda x:x[0])
            print("modify",f"{'_'.join(_name)}.parent",f"wecc240_psse_N_{_dn[0][1]};",file=_fh)
    return


@app.cell
def __():
    import os, sys
    import marimo as mo
    import pandas as pd
    from numpy import array
    sys.path.append("/usr/local/opt/gridlabd/current/share/gridlabd")
    import nsrdb_weather as ns
    return array, mo, ns, os, pd, sys


if __name__ == "__main__":
    app.run()
