"""Geodata sampling implementation

The `georecorder` sample data with geographic data included. Objects that lack
latitude and longitude data will inherit these from their parent objects.

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
            alias:dict={},
            virtual:dict={},
            **kwargs):
        """Create a georecorder

        Arguments:

        * `gridlabd`: the gridlabd module that links to the model

        * `csvname`: the CSV filename to record data

        * `objname`: the object name pattern to sample geodata from

        * `properties`: the property formatter dictionary

        * `alias`: the property column name alias dictionary (optional)

        * `virtual`: virtual property dictionary (name:[call,args])

        * `**kwargs`: options

        Options:

        * `include_latlon`: (bool) include lat/lon data (default False)
        """

        # options
        include_latlon = ("include_latlon" in kwargs and kwargs["include_latlon"] == True)

        # file handle
        self.fh = open(csvname,"w")

        # headers
        headers = ["datetime","geocode"]
        if include_latlon:
            headers += ["latitude","longitude"]
        if alias:
            headers += [(alias[x] if x in alias else x) for x in properties]
        else:
            headers += properties
        print(",".join(headers + list(virtual)),file=self.fh)

        # object specs
        self.objects = {}
        for obj in [x for x in gridlabd.get("objects") if re.match(objnames,x)]:
            getter = {x:gridlabd.property(obj,x) for x in properties.keys()}
            for name,values in virtual.values():
                for value in values:
                    getter[value] = gridlabd.property(obj,value)
            geocode = None
            parent = obj
            while parent is not None and geocode is None:
                data = gridlabd.get_object(parent)
                if "latitude" in data and "longitude" in data:
                    lat,lon = data["latitude"],data["longitude"]
                    geocode = {
                        "geocode":gridlabd.get_global(f"GEOCODE {lat},{lon}#6")
                    }
                    if include_latlon:
                        geocode["latitute"] = f"{lat:.6f}"
                        geocode["longitude"] = f"{lon:.6f}"
                else:
                    parent = data["parent"] if "parent" in data else None
            if not geocode:
                print(f"WARNING [{__name__}]: object '{obj}' has not geocode",file=sys.stderr)
            else:
                self.objects[obj] = RecorderSpec(geocode,getter)

        # properties
        self.properties = properties
        self.virtual = virtual

    def sample(self,t:int):
        """Sample data

        Arguments:

        * `t`: the timestamp (unix epoch)
        """
        for obj,recorder in self.objects.items():
            output = f"{dt.datetime.fromtimestamp(t)},{','.join(recorder.geocode.values())},"
            output += ",".join([str((call if call else lambda x:x)(recorder.getter[name].get_value())) for name,call in self.properties.items()])
            for call,values in self.virtual.values():
                args = {x:recorder.getter[x].get_value() for x in values}
                output += f",{call(args)}"
            print(output,file=self.fh)
