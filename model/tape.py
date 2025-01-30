"""Geodata sampling implementation

Example:

The following example samples load geodata from the WECC 240 model:

~~~
import tape

load_recorder = None
def on_init(t0):
    global load_recorder
    load_recorder = tape.GeoRecorder(gridlabd,"loads.csv","wecc240_psse_L",{
        "status" : None,
        "V" : lambda x: f"{abs(x):.3f}",
        "Vn": lambda x: f"{x:.3f}",
        "S" : lambda x: f"{abs(x):.1f}"
        })
    return True

def on_commit(t0):
    load_recorder.sample(t0)
    return True
~~~
"""

import sys
import re
import datetime as dt
from collections import namedtuple
from typing import TypeVar

RecorderSpec = namedtuple("RecorderSpec",["geocode","getter"])

class GeoRecorder:

    def __init__(self,gridlabd:TypeVar('module'),
            csvname:str,
            objnames:list[str],
            properties:dict,
            alias:dict={}):
        """Create a georecorder

        Arguments:

        * `gridlabd`: the gridlabd module that links to the model

        * `csvname`: the CSV filename to record data

        * `objname`: the object name pattern to sample geodata from

        * `properties`: the property getter dictionary

        * `alias`: the property column name alias dictionary (optional)
        """
        self.objects = {}
        self.fh = open(csvname,"w")
        self.properties = properties
        print(",".join(["datetime","geocode"]+list(properties)),file=self.fh)
        for obj in [x for x in gridlabd.get("objects") if re.match(objnames,x)]:
            getter = {x:gridlabd.property(obj,x) for x in properties.keys()}
            geocode = None
            parent = obj
            while parent is not None and geocode is None:
                data = gridlabd.get_object(parent)
                if "latitude" in data and "longitude" in data:
                    lat,lon = data["latitude"],data["longitude"]
                    geocode = gridlabd.get_global(f"GEOCODE {lat},{lon}#6")
                else:
                    parent = data["parent"] if "parent" in data else None
            if not geocode:
                print(f"WARNING [{__name__}]: object '{obj}' has not geocode",file=sys.stderr)
            else:
                self.objects[obj] = RecorderSpec(geocode,getter)

    def sample(self,t:int):
        """Sample data

        Arguments:

        * `t`: the timestamp (unix epoch)
        """
        for obj,recorder in self.objects.items():
            output = f"{dt.datetime.fromtimestamp(t)},{recorder.geocode},"
            output += ",".join([str((call if call else lambda x:x)(recorder.getter[name].get_value())) for name,call in self.properties.items()])
            print(output,file=self.fh)
