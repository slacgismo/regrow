"""Convert JSON to KML using mermaid

"""
import os
import sys
import datetime as dt
import json
import pandas as pd
from time import time

FILE = "wecc240.json"
VERBOSE = False
MAXRETRY = None

# Enable local mermaid server. (Actually just a passthru, so why bother. See
# https://hub.docker.com/r/jihchi/mermaid.ink to install local Mermaid
# server)

# os.putenv("MERMAID_INK_SERVER","http://localhost:3000")

import mermaid as md
import mermaid.graph as mg

with open(FILE,"r") as fh:
    model = json.load(fh)

objects = model["objects"]

names = pd.read_csv("wecc240_psse_bus.csv",index_col="id",dtype=str)
names.index=[str(x) for x in names.index]

def getlabel(name):
    label = name.split("_")[-1]
    if label in names.index:
        return f"{names.loc[label]['name']} ({label})"
    return label

def getname(name):
    label = name.split("_")[-1]
    if label in names.index:
        return names.loc[label]['name']
    return label

def getid(name,back=1):
    return "-".join(name.split("_")[-back:])

_cache = {}

def _decode(geohash):
    """
    Decode the geohash to its exact values, including the error
    margins of the result.  Returns four float values: latitude,
    longitude, the plus/minus error for latitude (as a positive
    number) and the plus/minus error for longitude (as a positive
    number).
    """
    __base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
    __decodemap = { }
    for i in range(len(__base32)):
        __decodemap[__base32[i]] = i
    del i
    lat_interval, lon_interval = (-90.0, 90.0), (-180.0, 180.0)
    lat_err, lon_err = 90.0, 180.0
    is_even = True
    for c in geohash:
        cd = __decodemap[c]
        for mask in [16, 8, 4, 2, 1]:
            if is_even: # adds longitude info
                lon_err /= 2
                if cd & mask:
                    lon_interval = ((lon_interval[0]+lon_interval[1])/2, lon_interval[1])
                else:
                    lon_interval = (lon_interval[0], (lon_interval[0]+lon_interval[1])/2)
            else:      # adds latitude info
                lat_err /= 2
                if cd & mask:
                    lat_interval = ((lat_interval[0]+lat_interval[1])/2, lat_interval[1])
                else:
                    lat_interval = (lat_interval[0], (lat_interval[0]+lat_interval[1])/2)
            is_even = not is_even
    lat = (lat_interval[0] + lat_interval[1]) / 2
    lon = (lon_interval[0] + lon_interval[1]) / 2
    return lat, lon, lat_err, lon_err

def geocode(geohash):
    """
    Decode geohash, returning two strings with latitude and longitude
    containing only relevant digits and with trailing zeroes removed.
    """
    global _cache
    if geohash in _cache:
        return _cache[geohash][0],_cache[geohash][1]
    lat, lon, lat_err, lon_err = _decode(geohash)
    from math import log10
    # Format to the number of decimals that are known
    lats = "%.*f" % (max(1, int(round(-log10(lat_err)))) - 1, lat)
    lons = "%.*f" % (max(1, int(round(-log10(lon_err)))) - 1, lon)
    if '.' in lats: lats = lats.rstrip('0')
    if '.' in lons: lons = lons.rstrip('0')
    _cache[geohash] = (float(lats), float(lons))
    return float(lats), float(lons)

def geohash(latitude, longitude, precision=6):
    """Encode a position given in float arguments latitude, longitude to
    a geohash which will have the character count precision.
    """
    from math import log10
    __base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
    __decodemap = { }
    for i in range(len(__base32)):
        __decodemap[__base32[i]] = i
    del i
    lat_interval, lon_interval = (-90.0, 90.0), (-180.0, 180.0)
    geohash = []
    bits = [ 16, 8, 4, 2, 1 ]
    bit = 0
    ch = 0
    even = True
    while len(geohash) < precision:
        if even:
            mid = (lon_interval[0] + lon_interval[1]) / 2
            if longitude > mid:
                ch |= bits[bit]
                lon_interval = (mid, lon_interval[1])
            else:
                lon_interval = (lon_interval[0], mid)
        else:
            mid = (lat_interval[0] + lat_interval[1]) / 2
            if latitude > mid:
                ch |= bits[bit]
                lat_interval = (mid, lat_interval[1])
            else:
                lat_interval = (lat_interval[0], mid)
        even = not even
        if bit < 4:
            bit += 1
        else:
            geohash += __base32[ch]
            bit = 0
            ch = 0
    result = ''.join(geohash)
    global _cache
    _cache[result] = (latitude,longitude)
    return result

network = {}
linecount = {}
nodenames = {}
def to_network(name,data):

    item = {}

    if data["class"] == "branch" or data["class"] == "dcline": 

        # branch object
        fbus,tbus = data["from"],data["to"]
        fkv,tkv = float(objects[fbus]["baseKV"].split()[0]),float(objects[tbus]["baseKV"].split()[0])
        geofrom = geohash(float(objects[fbus]["latitude"]),float(objects[fbus]["longitude"]))
        geoto = geohash(float(objects[tbus]["latitude"]),float(objects[tbus]["longitude"]))
        location = [geofrom,geoto]
        if geofrom != geoto:

            # powerline
            linecount[geofrom] = linecount[geofrom]+1 if geofrom in linecount else 1
            linecount[geoto] = linecount[geoto]+1 if geoto in linecount else 1
            item = {
                geofrom : f"{getid(tbus)}_{tkv:.0f}kV({geoto}) L{linecount[geofrom]}@-->|{fkv:.0f} kV| {getid(fbus)}_{fkv:.0f}kV[{fkv:.0f} kV]",
                geoto : f"{getid(fbus)}_{fkv:.0f}kV({geofrom}) L{linecount[geoto]}@-->|{fkv:.0f} kV| {getid(tbus)}_{tkv:.0f}kV[{tkv:.0f} kV]" 
            }

        else: # device

            if abs(fkv-tkv) > 1: # transformer, regulator, etc

                if fkv > tkv:
                    item = {geofrom:f"{getid(fbus)}_{fkv:.0f}kV[{fkv:.0f} kV] -->|OO| {getid(tbus)}_{tkv:.0f}kV[{tkv:.0f} kV]"}
                else:
                    item = {geofrom:f"{getid(tbus)}_{tkv:.0f}kV[{tkv:.0f} kV] -->|OO| {getid(fbus)}_{fkv:.0f}kV[{fkv:.0f}]"}

            else: # switch, etc

                item = {geofrom:f"{getid(fbus)}_{fkv:.0f}kV[{fkv:.0f} kV] -->|X| {getid(tbus)}_{tkv:.0f}kV[{tkv:.0f} kV]"}

    elif data["class"] == "bus": 

        # bus/load object
        location = geohash(float(data["latitude"]),float(data["longitude"]))
        global nodenames
        if not location in nodenames:
            nodenames[location] = []
        nodenames[location].append(getlabel(name))
        if "S" in data: # load
            S = complex(data['S'].split()[0])
            kv = float(data["baseKV"].split()[0])
            if abs(S) > 0:
                item = {location:f"""{getid(name)}_{kv:.0f}kV[{kv:.0f} kV] -->|{abs(S):,.1f} MVA| L_{getid(name)}@{{shape: tri, label: "{getid(name)}"}}"""}

    elif data["class"] == "gen": 

        pname = data["parent"]
        pdata = objects[pname]                                      
        location = geohash(float(pdata["latitude"]),float(pdata["longitude"]))
        pkv = float(pdata["baseKV"].split()[0])
        mva = abs(complex(float(data["Pg"].split()[0]),float(data["Qg"].split()[0])))
        item = {location:f"{getid(pname)}_{pkv:.0f}kV[{pkv:.0f} kV] -->|{-mva:,.1f} MVA| G_{getid(name,2)}(({getid(name,2)}))"}

    for location,spec in item.items():
        if not location in network:
            network[location] = []
        network[location].append(spec)

def to_graph(graph):
    if VERBOSE:
        tic = time()
        print("Mermaid graph:",flush=True,file=sys.stderr)
        print(graph,flush=True,file=sys.stderr)
    html = md.Mermaid(graph)._repr_html_()
    if html.startswith("Parse error"):
        print("ERROR [mermaid]:",graph)
        raise RuntimeError(html)
    if VERBOSE:
        print(html,flush=True,file=sys.stderr)
        print(f"Result in {time()-tic:.1f} seconds:",file=sys.stderr)
    return html

def print_bus(objects,file,container=None):

    generators = {x:y for x,y in objects.items() if y["class"] == "gen"}
    if not container is None:
        print(f"  <Folder><name>{container}</name>",file=file)

    print(f"Processing {container if container else 'all'} busses",end="",flush=True)
    for name,data in {x:y for x,y in objects.items() if y["class"] == "bus"}.items():
        label = getlabel(name)
        load = complex(data["S"].split()[0])
        if abs(load) > 0 and container in [None,"Loads"]:
            print(".",end="",flush=True)
            to_network(name,data)    

            print(f"""  <Placemark>
    <name>{getid(name)}</name>
    <styleUrl>#load</styleUrl>
    <description><table><caption>Load</caption>
        <tr><th>Voltage</th><td>{data["baseKV"]}</td></tr>
        <tr><th>Power</th><td>{abs(load):,.1f} MVA</td></tr></table>
    </description>
    <Point>
      <coordinates>{data["longitude"]},{data["latitude"]},0</coordinates>
    </Point>
  </Placemark>""",file=file)

        genlist = {x:y for x,y in generators.items() if y["parent"] == name}
        gen = sum([complex(float(y["Pg"].split()[0]),float(y["Qg"].split()[0])) for x,y in genlist.items()])
        if abs(gen) > 0 and container in [None,"Generators"]:
            print(".",end="",flush=True)
            for g,d in genlist.items():
                to_network(g,d)

                mva = complex(float(d["Pg"].split()[0]),float(d["Qg"].split()[0]))
                print(f"""  <Placemark>
    <name>{getid(g,2)}</name>
    <styleUrl>#gen</styleUrl>
    <description><table><caption>Generator</caption>
        <tr><th>Power</th><td>{abs(mva):,.1f} MVA</td></tr></table>
    </description>
    <Point>
      <coordinates>{data["longitude"]},{data["latitude"]},0</coordinates>
    </Point>
  </Placemark>""",file=file)

        if abs(gen) == 0 and abs(load) == 0 and container in [None,"Substations"]:
            print(".",end="",flush=True)
            to_network(name,data)

    if not container is None:
        print("</Folder>",file=file)
    print("ok",flush=True)

def print_branch(objects,file,container=None):

    if not container is None:
        print(f"  <Folder><name>{container}</name>",file=file)

    print(f"Processing {container if container else 'all'} branches",end="",flush=True)
    for name,data in {x:y for x,y in objects.items() if y["class"] in ["branch","dcline"]}.items():

        fbus = objects[data["from"]]
        tbus = objects[data["to"]]
        fromkv = float(fbus["baseKV"].split()[0])
        tokv = float(tbus["baseKV"].split()[0])

        if fromkv == tokv and container in [None,"Powerlines"]:
            print(".",end="",flush=True)
            to_network(name,data)
            print(f"""  <Placemark>
    <styleUrl>#powerline{"_down" if data["status"] != "IN" else int(fromkv/100)}</styleUrl>
    <name>{getlabel(data["from"])} --> {getlabel(data["to"])}</name>
    <LineString>
      <coordinates>
        {fbus["longitude"]},{fbus["latitude"]},50
        {tbus["longitude"]},{tbus["latitude"]},50
      </coordinates>
    </LineString>
  </Placemark>""",file=file)

        elif abs(fromkv - tokv) > 0.1 and container in [None,"Transformers"]:
            print(".",end="",flush=True)
            to_network(name,data)

    if not container is None:
        print("</Folder>",file=file)
    print("ok",flush=True)

def print_network(file,preamble=["graph TD"]):
    global nodenames
    print("Generating bus graphs",end="",flush=True)
    print("  <Folder><name>Nodes</name>",file=file)
    for node,graph in [(x,y) for x,y in network.items() if y]:
        print(".",end="",flush=True)
        graph += [f"L{n+1}@{{ curve: linear }}" for n in range(linecount[node] if node in linecount else 0)]
        lat,lon = geocode(node)
        svg = ""
        retry = 0
        graph = f"""---
title: {" / ".join(nodenames[node])}
---
""" + "\n  ".join(preamble+graph)
        while not svg.startswith("<svg "):
            if not MAXRETRY is None and retry >= MAXRETRY:
                print("ERROR:",svg,file=sys.stderr)
                print("SOURCE:",graph,file=sys.stderr,flush=True)
                raise RuntimeError("maximum mermaid graph retries")
            retry += 1
            try:
                svg = to_graph(graph)
            except Exception as err:
                print(f"WARNING: {err} (retrying)",file=sys.stderr,flush=True)

        print(f"""  <Placemark>
    <styleUrl>#bus</styleUrl>
    <name>{node}</name>
    <description>
        {svg.replace('width="100%"','width="640px" height="480px"')}
    </description>
    <Point>
      <coordinates>{lon},{lat},0</coordinates>
    </Point>
  </Placemark>""",file=file,flush=True)
    print("  </Folder>",file=file)
    print("ok",flush=True)

with open(os.path.splitext(FILE)[0]+".kml","w") as fh:

    loads = sum([abs(complex(y["S"].split()[0])) for x,y in objects.items() if y["class"] == "bus"])
    generators = sum([abs(complex(float(y["Pg"].split()[0]),float(y["Qg"].split()[0]))) for x,y in objects.items() if y["class"] == "gen" and y["status"] == "IN_SERVICE"])

    print(f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>

  <name>{FILE} ({dt.datetime.fromtimestamp(os.path.getmtime(FILE)).strftime("%Y-%m-%d %H:%M:%S")})</name>

  <description>
    Total load: {loads:,.1f} MVA
    Total generation: {generators:,.1f} MVA
  </description>

  <Style id="powerline0">
    <LineStyle>
      <color>7f00ffff</color>
      <width>1</width>
    </LineStyle>
    <PolyStyle>
      <color>7f00ffff</color>
    </PolyStyle>
  </Style>

  <Style id="powerline1">
    <LineStyle>
      <color>7f00ffff</color>
      <width>2</width>
    </LineStyle>
    <PolyStyle>
      <color>7f00ffff</color>
    </PolyStyle>
  </Style>

  <Style id="powerline2">
    <LineStyle>
      <color>7f00ffff</color>
      <width>3</width>
    </LineStyle>
    <PolyStyle>
      <color>7f00ffff</color>
    </PolyStyle>
  </Style>
  <Style id="powerline3">
    <LineStyle>
      <color>7f00ffff</color>
      <width>4</width>
    </LineStyle>
    <PolyStyle>
      <color>7f00ffff</color>
    </PolyStyle>
  </Style>
  <Style id="powerline4">
    <LineStyle>
      <color>7f00ffff</color>
      <width>5</width>
    </LineStyle>
    <PolyStyle>
      <color>7f00ffff</color>
    </PolyStyle>
  </Style>
  <Style id="powerline5">
    <LineStyle>
      <color>7f00ffff</color>
      <width>6</width>
    </LineStyle>
    <PolyStyle>
      <color>7f00ffff</color>
    </PolyStyle>
  </Style>

  <Style id="powerline_down">
    <LineStyle>
      <color>7f000000</color>
      <width>4</width>
    </LineStyle>
    <PolyStyle>
      <color>7f000000</color>
    </PolyStyle>
  </Style>

  <Style id="bus">
    <IconStyle>
      <scale>1.0</scale>
      <Icon>
        <href>https://icons.veryicon.com/png/o/business/project-3/transformer-substation-2.png</href>
      </Icon>
    </IconStyle>
  </Style>

  <Style id="load">
    <IconStyle>
      <scale>1.0</scale>
      <Icon>
        <href>https://cdn-icons-png.flaticon.com/128/3061/3061341.png</href>
      </Icon>
    </IconStyle>
  </Style>

  <Style id="gen">
    <IconStyle>
      <scale>1.0</scale>
      <Icon>
        <href>https://icons.veryicon.com/png/o/commerce-shopping/flat-icons-for-business-and-finance/power-plant.png</href>
      </Icon>
    </IconStyle>
  </Style>

  <Style id="transformer">
    <IconStyle>
      <scale>1.0</scale>
      <Icon>
        <href>https://icons.veryicon.com/png/o/miscellaneous/vertical-menu/transformer-7.png</href>
      </Icon>
    </IconStyle>
  </Style>

  <Style id="shunt">
    <IconStyle>
      <scale>1.0</scale>
      <Icon>
        <href>https://icons.veryicon.com/png/o/education-technology/power-icon-2/shunt-reactor.png</href>
      </Icon>
    </IconStyle>
  </Style>
""",file=fh)

    print_bus(objects,file=fh,container="Substations")
    print_bus(objects,file=fh,container="Generators")
    print_bus(objects,file=fh,container="Loads")
    print_branch(objects,file=fh,container="Powerlines")
    print_branch(objects,file=fh,container="Transformers")

    print_network(file=fh)

    print("</Document></kml>",file=fh)

