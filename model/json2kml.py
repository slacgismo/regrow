
import os
import datetime as dt
import json
import pandas as pd

FILE = "wecc240.json"

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

def print_bus(objects,file,container=None):

    generators = {x:y for x,y in objects.items() if y["class"] == "gen"}
    if not container is None:
        print(f"  <Folder><name>{container}</name>",file=file)

    for name,data in {x:y for x,y in objects.items() if y["class"] == "bus"}.items():
        
        label = getlabel(name)

        load = complex(data["S"].split()[0])
        if abs(load) > 0 and container in [None,"Loads"]:
        
            print(f"""  <Placemark>
    <name>{label}</name>
    <styleUrl>#load</styleUrl>
    <description><table><caption>Load</caption>
        <tr><th>Voltage</th><td>{data["baseKV"]}</td></tr>
        <tr><th>Power</th><td>{abs(load):,.1f} MVA</td></tr></table>
    </description>
    <Point>
      <coordinates>{data["longitude"]},{data["latitude"]},0</coordinates>
    </Point>
  </Placemark>""",file=file)

        gen = sum([complex(float(y["Pg"].split()[0]),float(y["Qg"].split()[0])) for x,y in generators.items() if y["parent"] == name and y["status"] == "IN_SERVICE"])
        if abs(gen) > 0 and container in [None,"Generators"]:
        
            print(f"""  <Placemark>
    <name>{label}</name>
    <styleUrl>#gen</styleUrl>
    <description><table><caption>Generator</caption>
        <tr><th>Voltage</th><td>{data["baseKV"]}</td></tr>
        <tr><th>Power</th><td>{abs(gen):,.1f} MVA</td></tr></table>
    </description>
    <Point>
      <coordinates>{data["longitude"]},{data["latitude"]},0</coordinates>
    </Point>
  </Placemark>""",file=file)

        if abs(gen) == 0 and abs(load) == 0 and container in [None,"Substations"]:
            print(f"""  <Placemark>
    <name>{label}</name>
    <styleUrl>#bus</styleUrl>
    <description><table><caption>Bus</caption>
        <tr><th>Voltage</th><td>{data["baseKV"]}</td></tr>
        <tr><th>Bustype</th><td>{data["type"]}</td></tr></table>
    </description>
    <Point>
      <coordinates>{data["longitude"]},{data["latitude"]},0</coordinates>
    </Point>
  </Placemark>""",file=file)

    if not container is None:
        print("</Folder>",file=file)


def print_branch(objects,file,container=None):

    if not container is None:
        print(f"  <Folder><name>{container}</name>",file=file)

    for name,data in {x:y for x,y in objects.items() if y["class"] == "branch"}.items():
        fbus = objects[data["from"]]
        tbus = objects[data["to"]]
        fromkv = float(fbus["baseKV"].split()[0])
        tokv = float(tbus["baseKV"].split()[0])

        if fromkv == tokv and container in [None,"Powerlines"]:
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
            print(f"""  <Placemark>
    <styleUrl>#transformer</styleUrl>
    <name>{getlabel(data["from"])} --> {getlabel(data["to"])}</name>
    <description><table><caption>Transformer</caption>
        <tr><th>From</th><td>{fbus["baseKV"]} ({fbus["class"]} {data["from"].split("_")[-1]})</td></tr>
        <tr><th>To</th><td>{tbus["baseKV"]} ({tbus["class"]} {data["to"].split("_")[-1]})</td></tr></table>
    </description>
    <Point>
      <coordinates>{fbus["longitude"]},{fbus["latitude"]},0</coordinates>
    </Point>
  </Placemark>""",file=file)

    if not container is None:
        print("</Folder>",file=file)


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

    print("</Document></kml>",file=fh)