"""GridLAB-D optimal powerflow/sizing/placement

Example:

The following example loads the 4-bus model and attempts an OPF. However,
there is insufficient generation to avoid curtailment. Then it runs
the optimal sizing/placement problem and updates the model with the result.
Then the OPF runs without curtailment and the simulation is run with the new model.

>>> import gld
>>> test = Model("test.json")
>>> test.optimal_powerflow()["curtailment"]
>>> test.optimal_sizing(gen_cost=np.array([100,500,1000,1000])+1000j,
                        cap_cost={0:1000,1:500},
                        update_model=True)
>>> test.optimal_powerflow(refresh=True)["curtailment"]
>>> test.run("test_out.json")
"""

import sys
import os
import json
import math
import io
import subprocess
import numpy as np
try:
    from numpy import Inf
except:
    np.Inf = np.inf
import numpy.linalg as la
import cvxpy as cp
from typing import Union, Any, TypeVar
import warnings
try:
    from pypower.api import runpf, runopf, ppoption, printpf
except ModuleNotFoundError as err:
    def pypower_api(*args,**kwargs):
        raise RuntimeError(f"pypower not available ({err})")
    runpf = runopf = ppoption = printpf = pypower_api

np.set_printoptions(linewidth=np.inf,formatter={float:lambda x:f"{x:8.4f}"})

class Model:
    """GridLAB-D model handler"""

    # format rules
    formatter = {
        float: lambda x: f"{x:8.2f}",
        complex: lambda x: f"{x.real:8.2f}{x.imag:+8.2f}",
        int: lambda x: f"{x:8.0f}",
    }

    def __init__(self,data):
        if isinstance(data,str):
            if data[0] == '{':
                data = json.loads(data)
            else:
                data = json.load(open(data,"r"))
        elif isinstance(data,io.StringIO):
            data = json.load(data)
        if not isinstance(data,dict):
            raise TypeError("data is not a dict, filename, or JSON string")
        self.name = data["globals"]["modelname"]["value"]
        self.data = data
        self.results = {}
        self.modified = False
        self._last_name = None
        self._last_data = None

    def __repr__(self):
        return f"Model({repr(self.name)})"

    def validate(self,modules:str=[]):
        """Validate a GridLAB-D model

        Arguments:
        * model: the GridLAB-D model
        * modules: list of modules that must be present in the model 

        Returns:
        * None: model contain no errors
        * str: model contains an error
        """
        if "application" not in self.data:
            return "model does not contain application name"
        if self.data["application"] != "gridlabd":
            return "model is not from a gridlabd application"
        for module in modules:
            if module not in self.data["modules"]:
                return f"model does not contain module {module}"

    def modules(self) -> list[str]:
        """Return list of active modules"""
        return list(self.data["modules"])

    def classes(self,module:str=None,astype:type=list):
        """Get classes

        Arguments:
        * module: name of module (None return all classes)
        * astype: format of result

        Returns:
        * astype: result
        """
        if module:
            astype(self.data["classes"])
        return astype(self.data["classes"][module])

    def get_property(self,obj:str,name:str|list,astype:str=None) -> Any:
        """Get an object property and convert to Python type

        Arguments:
        * obj: name of object
        * name: name of property
        * astype: force property to type

        Returns:
        * varies: value of object property
        """
        if isinstance(obj,list):
            return {x:self.get_property(x,name,astype) for x in obj}
        if isinstance(name,list):
            return [self.get_property(obj,x,astype) for x in name]
        if obj != self._last_name:
            object_data = self.data["objects"][obj]
            self._last_name = obj
            self._last_data = object_data
        else:
            object_data = self._last_data
        if name in self.data["header"]:
            ptype = self.data["header"][name]["type"]
            if ptype in dir(self):
                result = getattr(self,ptype)(object_data[name])
            else:
                result = object_data[name]
        else:
            ptype = self.data["classes"][object_data["class"]][name]["type"]
            if ptype in dir(self):
                result = getattr(self,ptype)(object_data[name])
            else:
                result = object_data[name]

        if astype is None or isinstance(result,astype):
            return result
        elif name in self.data["header"]:
            spec = self.data["header"][name]
        elif name in self.data["classes"][object_data["class"]]:
            spec = self.data["classes"][object_data["class"]][name]
        else:
            return astype(result)

        if astype in [int,float] and isinstance(spec,dict) and spec["type"] == "enumeration":

            return astype(int(spec["keywords"][result][2:],16))

        elif astype is int and isinstance(spec,dict) and spec["type"] == "set":

            raise NotImplementedError("set astype")

        return astype(result)

    def set_property(self,obj,**kwargs):

        for name,value in kwargs.items():
            if name in self.data["objects"]:
                self.data["objects"][obj][name] = type(value)
        return self.data["objects"][obj]            

    def format(self,value:Any) -> str:
        """Apply formatting rules

        Arguments:
        * value: value to format

        Returns:
        * str: formatted value

        See `formatter`.
        """
        try:
            return self.formatter[type(value)](value)
        except:
            return str(value)

    @staticmethod
    def double(x:str) -> float:
        """Extract a double value from a GridLAB-D property"""
        return float(x.split()[0])

    @staticmethod
    def complex(x:str) -> complex:
        """Extract a complex value from a GridLAB-D property"""
        return complex(x.split()[0])

    @staticmethod
    def unit(x:str) -> str:
        """Extract a unit specifier from a GridLAB-D property"""
        return x.split()[1]

    @staticmethod
    def int64(x:str) -> int:
        """Extract an integer value from a GridLAB-D property"""
        return int(x)

    @staticmethod
    def int32(x:str) -> int:
        """Extract an integer value from a GridLAB-D property"""
        return int(x)

    @staticmethod
    def int16(x:str) -> int:
        """Extract an integer value from a GridLAB-D property"""
        return int(x)

    @staticmethod
    def int8(x:str) -> int:
        """Extract an integer value from a GridLAB-D property"""
        return int(x)

    @staticmethod
    def bool(x:str) -> bool:
        """Extract a boolean value from a GridLAB-D property"""
        return {"FALSE":False,"TRUE":True}[x]

    def find(self,
             oclass:str,
             astype:type=dict,
            ) -> dict|list:
        """Find objects of a class

        Arguments:
        * model: the GridLAB-D model
        * oclass: the desired class
        * astype: the return value (list or dict)

        Returns:
        * list: list of object names
        * dict: object data by name
        """
        if astype == list:
            return [x for x, y in self.data["objects"].items() if y["class"] == oclass]
        elif astype == dict:
            return {x:y for x, y in self.data["objects"].items() if y["class"] == oclass}
        raise ValueError("astype is not valid")

    def select(self,
               criteria:dict,
               startwith:dict = None
              ) -> dict:
        """Select objects matching criteria

        Arguments:
        * criteria: selection criteria as "property":"value" dictionary
        * startwith: data dictionary to start with

        Returns:
        * dict: object data by name
        """
        if startwith is None:
            startwith = self.data["objects"]
        result = dict(startwith)
        for key,value in criteria.items():
            result = {x:y for x,y in result.items() if key in y and y[key] == value}
        return result


    def globals(self,name:str=None) -> Union[bool,int,float,complex,str]:
        """Get global variables

        Arguments:
        * name: name of global variable

        Returns:
        * type: value of global variable of type according to GridLAB-D type
        """
        if name == list:
            return list(self.data["globals"])
        if name == dict:
            return {x:self.globals(x) for x,y in self.data["globals"].items()}
        spec = self.data["globals"][name]
        astype = getattr(self,spec["type"]) if hasattr(self,spec["type"]) else str
        return astype(self.data["globals"][name]["value"])

    def get_result(self,name:str) -> Any:
        """Get result from cache

        Arguments:
        * name: cache variable name

        Returns: 
        * varies: cached result if any
        """
        return self.results[name]

    def set_result(self,name,value:Any) -> Any:
        """Set result in cache

        Arguments:
        * name: cache variable name
        * value: value to stored in cache

        Returns:
        * varies: value stored in cache
        """
        self.results[name] = value
        return value

    #
    # Model handling
    #
    def get_object(self,obj:str) -> dict:
        """Get object data

        Arguments:
        * obj: object name

        Returns:
        * dict: object data
        """
        return self.data["objects"][obj]

    def add_object(self,oclass:str,obj:str,**kwargs) -> dict:
        """Add object

        Arguments:
        * oclass: object class
        * obj: object name
        * kwargs: object data

        Returns:
        * dict: object data
        """
        if oclass not in self.data["classes"]:
            raise ValueError(f"class '{oclass}' not found")
        if obj in self.data["objects"]:
            raise ValueError(f"object '{obj}' already defined")
        if "id" in kwargs and int(kwargs["id"]) in [int(x["id"]) for x in self.data["objects"].values()]:
            raise ValueError(f"id '{kwargs['id']}' is already used")
        data = {"class":oclass,"id":str(max([int(x["id"]) for x in self.data["objects"].values()])+1)}
        classdata = self.data["classes"][oclass]
        for name,value in kwargs.items():
            if name not in classdata and name not in self.data["header"]:
                raise ValueError(f"property '{name}' not valid in class '{oclass}'")
            if not isinstance(value,str):
                if name in self.data["header"]:
                    data[name] = str(value)
                elif "unit" in classdata[name]:
                    data[name] = f"{value} {classdata[name]['unit']}"
                elif classdata[name]["type"] == "bool":
                    data[name] = "TRUE" if value else "FALSE"
                else:
                    data[name] = str(value)
            else:
                data[name] = value
        for name,spec in classdata.items():
            if name not in data:
                if "flags" in spec and "REQUIRED" in spec["flags"].split("|"):
                    raise ValueError(f"property '{name}' is required")
                if "default" in spec:
                    data[name] = spec["default"]
        self.modified = True
        self.results = {}
        self.data["objects"][obj] = data
        return data

    def del_object(self,obj,on_ref='error',on_error='ignore'):
        if isinstance(obj,list):
            return [del_object(x,on_ref) for x in obj]
        if obj not in self.data["objects"] and on_error != 'ignore':
            raise ValueError(f"object '{obj}' not found")
        result = self.data["objects"][obj]
        if not on_ref == 'ignore':
            found = []
            for name,data in result.items():
                if "parent" in data and data["parent"]:
                    found.append(data["parent"])
                oclass = self.data["classes"][result["class"]]
                for prop,value in [(x,y) for x,y in result.items() if x not in self.data["header"]]:
                    if oclass[prop]["type"] == "object" and value == obj:
                        found.append(value)
            if found:
                if on_ref == 'delete':
                    for item,data in del_object(found,on_ref):
                        result[item] = data
                elif on_ref == 'error' and on_error != 'ignore':
                    raise RuntimeError("object is referenced by another object")
        del self.data["objects"][obj]
        return result

    def mod_object(self,obj,**kwargs):

        if not obj in self.data["objects"]:
            raise ValueError(f"object '{obj}' not defined")
        if "id" in kwargs and int(kwargs["id"]) in [int(x["id"]) for x in self.data["objects"].values()]:
            raise ValueError(f"id '{kwargs['id']}' is already used")
        data = self.data["objects"][obj]
        classdata = self.data["classes"][data["class"]]
        for name,value in kwargs.items():
            if "name" in data and data["name"] != name:
                raise ValueError(f"object name '{name}' does not match data['name']")
            if name not in classdata and name not in self.data["header"]:
                raise ValueError(f"property '{name}' not valid in class '{oclass}'")
            if value is None:
                if name in ["class","name"]:
                    raise ValueError(f"property '{name}' cannot be deleted")
                del data[value]
            elif not isinstance(value,str):
                if name in self.data["header"]:
                    data[name] = str(value)
                elif "unit" in classdata[name]:
                    data[name] = f"{value} {classdata[name]['unit']}"
                elif classdata[name]["type"] == "bool":
                    data[name] = "TRUE" if value else "FALSE"
                else:
                    data[name] = str(value)
            else:
                data[name] = value
        for name,spec in classdata.items():
            if name not in data:
                if "flags" in spec and "REQUIRED" in spec["flags"].split("|"):
                    raise ValueError(f"property '{name}' is required")
                if "default" in spec:
                    data[name] = spec["default"]
        self.modified = True
        self.results = {}
        self.data["objects"][obj] = data
        return data

    def save(self,name=None,**kwargs):

        with open(name if name else self.name,"w") as fh:
            json.dump(self.data,fh,**kwargs)
        self.modified = False

    def run(self,name=None,binary="GLD_ETC" in os.environ,exception=True,*args,**kwargs):
        if self.modified:
            raise RuntimeError("model has been modified")
        if not name:
            name = self.name
        command = ["gridlabd.bin" if binary else "gridlabd"]
        if name:
            command.append(name)
        command.extend(args)
        for x,y in kwargs.items():
            command.extend(["-D",f"{x}={y}"])
        result = subprocess.run(command,capture_output=True)
        status = result.returncode
        stdout = result.stdout.decode("utf-8").strip()
        stderr = result.stderr.decode("utf-8").strip()
        if exception:
            if status != 0:
                raise RuntimeError(stderr)
            else:
                return stdout
        return status,stdout.split('\n'),stderr.split('\n')

    #
    # PyPOWER support
    #
    def assert_module(self,name:str):
        """Assert that pypower module is found"""
        assert name in self.data["modules"], f"{name} module not found"

    def get_name(self,kind:str,id:int|list[int]=None) -> list:
        """Get bus/branch name

        Arguments:
        * kind: 'bus' or 'branch'
        * id: bus/branch index

        Returns:
        * str: name of bus/branch at index id
        * list[str]: list of names of busses/branches at indexes id
        """
        if isinstance(id,int):
            return self.find(kind,list)[id]
        elif isinstance(id,list):
            return [x for n,x in enumerate(self.find(kind,list)) if n in id]
        elif id is None:
            return self.find(kind,list)
        else:
            raise TypeError("id must be an int, list, or None")

    def get_bus(self,name:str|list) -> str|list:
        """Get bus name

        Arguments:
        * name: object name

        Returns:
        * int: bus id for specified object name
        * list: bus names for specified object names
        """
        if isinstance(name,list):
            return [self.get_bus(x) for x in name]
        if self.get_property(name,"class") == "bus":
            return name
        parent = self.get_property(name,"parent")
        return self.get_bus(parent)

    def get_branch(self,kind:str,id:int|list[int]=None):
        return None

    def get_areas(self) -> list:
        return list(set([x["area"] for x in self.find("bus",dict).values()]))

    def perunit(self,kind:str,refresh:bool=True) -> Union[list,float]:
        """Get the per-unit values in the pypower model

        Arguments:
        """
        self.assert_module("pypower")
        if "perunit"+kind in self.results:
            return self.results["perunit"+kind]
        elif kind == 'S':
            self.results["perunit"+kind] = self.globals("pypower::baseMVA")
        elif kind == 'V':
            self.results["perunit"+kind] = [self.get_property(x,"baseKV") for x in self.find("bus")]
        elif kind == 'Z':
            names = {y["bus_i"]:x for x,y in self.find("bus").items()}
            self.results["perunit"+kind] = [self.get_property(names[x["fbus"]],"baseKV")**2/self.globals("pypower::baseMVA") for x in self.find("branch",dict).values()]
        else:
            raise ValueError("invalid kind")
        return self.results["perunit"+kind]

    def lines(self,refresh:bool=False) -> dict:
        """Get line data

        Arguments:
        * refresh: force regeneration of data from model

        Returns:
        * dict: line data
        """
        if "lines" in self.results and not refresh:
            return self.results["lines"]
        self.assert_module("pypower")
        if "lines" in self.results:
            return self.results["lines"]
        self.results["lines"] = self.find("branch",dict)
        return self.results["lines"]

    def nodes(self,refresh:bool=False) -> dict:
        """Get node data
        
        Arguments:
        * refresh: force regeneration of data from model

        Returns:
        * dict: node data
        """
        if "nodes" in self.results and not refresh:
            return self.results["nodes"]
        self.assert_module("pypower")
        if "nodes" in self.results:
            return self.results["nodes"]
        self.results["nodes"] = self.find("bus")
        return self.results["nodes"]

    def generators(self,refresh:bool=False) -> dict:
        """Get generator data

        Arguments:
        * refresh: force regeneration of data from model

        Returns:
        * dict: generator data
        """
        if "generators" in self.results and not refresh:
            return self.results["generators"]
        self.assert_module("pypower")
        if "generators" in self.results:
            return self.results["generators"]
        self.results["generators"] = self.find("gen",dict)
        return self.results["generators"]

    def costs(self,refresh:bool=False) -> dict:
        """Get generation cost data

        Arguments:
        * refresh: force regeneration of data from model

        Returns:
        * dict: cost data
        """
        if "costs" in self.results and not refresh:
            return self.results["costs"]
        self.assert_module("pypower")
        if "costs" in self.results:
            return self.results["costs"]
        self.results["costs"] = self.find("gencost")
        return self.results["costs"]

    def impedance(self,refresh:bool=False) -> np.array:
        """Get impedance array

        Arguments:
        * refresh: force regeneration of data from model

        Returns:
        * np.array: line impedance array
        """
        self.assert_module("pypower")
        if "impedance" in self.results and not refresh:
            return self.results["impedance"]
        self.results["impedance"] = [complex(self.get_property(x,"r"),self.get_property(x,"x")) for x in self.lines(refresh)]
        return self.results["impedance"]

    def graphLaplacian(self,refresh:bool=False) -> np.array:
        """Get network graph Laplacian

        Arguments:
        * refresh: force recalculation of previous results

        Returns:
        * np.array: graph Laplacian matrix
        """
        if "graphLaplacian" in self.results and not refresh:
            return self.results["graphLaplacian"]
        self.assert_module("pypower")
        lines = self.lines(refresh)
        N = len(self.nodes(refresh))
        B = np.array([[int(x["fbus"])-1,int(x["tbus"])-1] for x in lines.values()])
        R = self.impedance(refresh)
        G = np.zeros((N,N),dtype=complex)
        for n, l in enumerate(B):
            G[l[0], l[1]] = G[l[1], l[0]] = ( 1 / R[n] ) if abs(R[n]) > 0 else 1e6
        self.results["graphLaplacian"] = np.diag(sum(G)) - G # graph Laplacian
        return self.results["graphLaplacian"]

    def graphIncidence(self,refresh:bool=False,weighted:bool=True) -> np.array:
        """Get network indicidence matrix

        Arguments:
        * refresh: force recalculation of previous results

        Returns:
        * np.array: incidence matrix
        """
        cachename = f"graphIncidence.{'weighted' if weighted else 'unweighted'}"
        if cachename in self.results and not refresh:
            return self.results[cachename]
        self.assert_module("pypower")
        lines = self.find("branch")
        N = len(self.find("bus",list))
        L = len(lines)
        B = np.array([[int(x["fbus"])-1,int(x["tbus"])-1] for x in lines.values()])
        R = [self.get_property(x,"r")+self.get_property(x,"x")*1j for x in lines] if weighted else np.ones(L,dtype=complex)
        I = np.zeros((L, N))  # link-node incidence matrix
        for n, l in enumerate(B):
            I[n][l[0]] = -R[n].real
            I[n][l[1]] = R[n].real
        self.results[cachename] = I
        return self.results[cachename]

    def graphSpectral(self,refresh:bool=False) -> tuple[float]:
        """Get spectral analysis results

        Arguments:
        * refresh: force recalculation of previous results

        Returns:
        * tuple: (E,U,K) where E is the eigenvalues, U is the eigenvectors,
          and K is the number of networks found
        """
        self.assert_module("pypower")
        if "graphSpectral" in self.results and not refresh:
            return self.results["graphSpectral"]
        G = self.graphLaplacian(refresh)
        e,u = la.eig(G)
        i = e.argsort()
        E,U = np.abs(e[i].round(6)),u.T[i]
        K = sum([1 if x==0 else 0 for x in E])
        self.results["graphSpectral"] = (E,U,K)
        return self.results["graphSpectral"]

    def demand(self,kind:str='actual',refresh:bool=False) -> np.array:
        """Get demand array

        Arguments:
        * kind: 'actual' or 'peak' demand
        * refresh: force recalculation of previous results

        Returns:
        * np.array: demand vector
        """
        if f"demand.{kind}" in self.results and not refresh:
            return self.results[f"demand.{kind}" ]
        if kind == "actual":
            self.results[f"demand.{kind}" ] = np.array([complex(self.get_property(x,"Pd"),self.get_property(x,"Qd")) for x in self.nodes(refresh)]) / self.perunit("S")
        elif kind == "peak":
            self.results[f"demand.{kind}" ] = np.array([complex(self.get_property(x,"Pd"),self.get_property(x,"Qd")) for x in self.nodes(refresh)]) / self.perunit("S")
        else:
            raise ValueError(f"kind '{kind}' is invalid")
        return self.results[f"demand.{kind}" ]

    def generation(self,kind:str='capacity',refresh:bool=False) -> np.array:
        """Get generation array

        Arguments:
        * kind: 'actual' or 'capacity'
        * refresh: force recalculation of previous results

        Returns:
        * np.array: generation vector
        """
        try:
            if not refresh:
                return self.get_result(f"generators.{kind}")
        except:
            pass
        puS = self.perunit("S",refresh)
        if kind == 'capacity':
            gen = [(self.get_property(x,"bus"),complex(self.get_property(x,"Pmax")/puS,self.get_property(x,"Qmax")/puS)) for x in self.generators(refresh)]
        elif kind == 'actual':
            gen = [(self.get_property(x,"bus"),complex(self.get_property(x,"Pg")/puS,self.get_property(x,"Qg")/puS)) for x in self.generators(refresh)]
        else:
            raise ValueError(f"kind '{kind}' is invalid")
        result = np.zeros(len(self.nodes(refresh)),dtype=complex)
        for bus,value in gen:
            result[bus-1] += value
        return self.set_result(f"generators.{kind}",result)

    def prices(self,refresh:bool=False) -> np.array:
        """Get generation price array

        Arguments:
        * refresh: force recalculation of previous results

        Returns:
        * np.array: price vector
        """
        if "prices" in self.results and not refresh:
            if not refresh:
                return self.results["prices"]
        costs = {self.get_property(y["parent"],"bus"):float(y["costs"].split(",")[1]) for x,y in self.costs(refresh).items()}
        self.results[f"prices"] = np.array([costs[n+1] if n+1 in costs else 0 for n in range(len(self.nodes(refresh)))])
        return self.results[f"prices"]

    def shunts(self,refresh:bool=False) -> dict:
        """Get shunt device data

        Arguments:
        * refresh: force regeneration of data from original model

        Returns:
        * dict: shunt device data
        """
        try:
            if not refresh:
                return self.get_result(f"shunts")
        except:
            pass
        result = {y["parent"]:{'setting':self.get_property(x,"admittance"),'capacity':self.get_property(x,"admittance_1")*self.get_property(x,"steps_1")} for x,y in self.find("shunt").items()}
        return self.set_result(f"shunts",result)

    def capacitors(self,kind:str='installed',refresh:bool=False) -> np.array:
        """Get capacitor array

        Arguments:
        * kind: 'installed' or 'setting'
        * refresh: force recalculation of previous results

        Returns:
        * np.array: capacitor array
        """
        try:
            if not refresh:
                return self.get_result(f"capacitors.{kind}")
        except:
            pass
        puS = self.perunit("S",refresh)
        shunts = self.shunts(refresh)
        if kind == 'installed':
            cap = [(self.get_property(x,"bus_i"),shunts[x]["capacity"]/puS if x in shunts else 0.0) for x in self.nodes(refresh)]
        elif kind == 'setting':
            cap = [(self.get_property(x,"bus_i"),shunts[x]["setting"]/puS if x in shunts else 0.0) for x in self.nodes(refresh)]
        else:
            raise ValueError(f"kind '{kind}' is invalid")
        result = np.zeros(len(self.nodes(refresh)))
        for bus,value in cap:
            result[bus-1] += value
        return self.set_result(f"capacitors.{kind}",result)

    def lineratings(self,rating:str="A",refresh:bool=False) -> np.array:
        """Get line ratings array

        Arguments:
        * rating: 'A', 'B', or 'C'
        * refresh: force recalculation of previous results

        Returns:
        * np.array: array of line ratings
        """
        if f"lineratings.{rating}" in self.results and not refresh:
            return self.results[f"lineratings.{rating}"]
        if rating not in "ABC":
            return ValueError(f"line rating '{rating}' is invalid'")
        self.results[f"lineratings.{rating}"] = np.array([self.get_property(x,f"rate{rating}") for x in self.lines()])/self.perunit('S',refresh) 
        return self.results[f"lineratings.{rating}"]

    def lineflow(self,refresh:bool=False) -> np.array:
        """Get line flows

        Arguments:
        * refresh: force recalculation of previous results

        Returns:
        * np.array: line flows pu.MVA
        """
        I = self.graphIncidence(refresh=refresh,weighted=True)
        x = [self.get_property(x,'Va') for x in self.find('bus')]
        return I@x

    def linevoltage(self,part:str='Va',refresh:bool=False) -> np.array:
        """Get line voltage angles/magnitude differences

        Arguments:
        * part: 'Va' or 'Vm'
        * refresh: force recalculation of previous results

        Returns:
        * np.array: line voltage angle/magnitude differences
        """
        I = self.graphIncidence(refresh=refresh,weighted=False)
        x = [self.get_property(x,part) for x in self.find('bus')]
        return I@x

    def linesplit(self,angle_limit:float=10.0,update_model=False) -> dict:
        """Identify/fix lines with large voltage angles

        Arguments:
        * angle_limit: the maximum angle allowed before a line is split

        Returns:
        * dict: line names and angles
        """
        result = {}

        # compute voltage angle differences
        I = self.graphIncidence(weighted=False)
        x = [self.get_property(x,'Va') for x in self.find('bus')]
        Va = I@x

        # identify large angles
        for n,v in enumerate(Va.tolist()):
            if abs(v) > angle_limit:
                result[self.get_name('branch',n)] = v
        if not update_model:
            return result

        # update model
        for name,angle in result.items():
            n = int(abs(angle)//angle_limit) + 1
            print("\nsplitting line",name,"in",n,"parts",file=sys.stderr)
            fbus,tbus = self.get_property(name,["fbus","tbus"])
            data = self.get_object(name)
            values = {
                "r": self.get_property(name,"r")/n,
                "x": self.get_property(name,"x")/n,
                "b": self.get_property(name,"b")*n,
                "angle": self.get_property(name,"angle")/n,
                "ratio": self.get_property(name,"ratio")**(1/n),
                "loss": self.get_property(name,"loss")/n,
            }
            # print("before",data,file=sys.stderr)
            self.set_property(name,**values)
            print("after",data,file=sys.stderr)
            # self.mod_object(name,data)
            

        return result

    #
    # Optimizations
    #
    def _solver_failed(err):
        """Failed solution default handler"""
        raise RuntimeError(err)

    def _problem_invalid(err):
        """Invalid problem default handler"""
        raise ValueError(err)

    def optimal_powerflow(self,
        refresh:bool=False,
        verbose:bool|TypeVar('io.TextIOWrapper')=False,
        curtailment_price=None,
        ref:int|str = None,
        angle_limit:float=10.0,
        voltage_limit:float=0.05,
        on_invalid:callable=_problem_invalid,
        on_fail:callable=_solver_failed,
        **kwargs) -> dict:
        """Compute optimal powerflow

        Arguments:
        * refresh: force recalculation of previous result
        * verbose: output solver data and results
        * curtailment_price: price at which load is curtailed
        * ref: reference bus id or name
        * angle_limit: voltage angle accuracy limit
        * voltage_limit: voltage magnitude violation limit
        * on_invalid: invalid problem handler
        * on_fail: solution failed handler
        * kwargs: options passed of cvxpy.Problem.solve()

        Returns:
        * dict: solution results
        """
        try:
            if not refresh and not self.modified:
                return self.get_result("optimal_powerflow")
        except:
            pass

        if self.graphSpectral()[2] > 1:
            return on_invalid(f"{self.name} cannot solve OPF on more than one network at a time (model has {self.graphSpectral()[2]} networks)")
        if self.graphSpectral()[2] == 0:
            return on_invalid(f"{self.name} cannot solve OPF on invalid network modles (no zero eigenvalues found)")

        # setup verbose output
        if verbose is True:
            verbose = sys.stderr

        # extract data from model   
        try:
            if ref is None:
                ref = self.select({"class":"bus","type":"REF"})
                if len(ref) > 0:
                    if len(ref) > 1:
                        warnings.warn(f"{self.name} multiple reference busses found {ref}, using bus {ref[0]}")
                    ref = ref[list(ref)[0]]["bus_i"]
                else:
                    warnings.warn(f"{self.name} no reference bus found, using bus 0")
                    ref = 0
            elif isinstance(ref,str):
                ref = self.get_bus(ref)
            P = self.prices(refresh)
            G = self.graphLaplacian(refresh)
            D = self.demand('actual',refresh)
            I = self.graphIncidence(refresh)
            F = self.lineratings("A",refresh)
            S = self.generation('capacity',refresh)
            C = self.capacitors('installed',refresh)
            N = len(self.nodes(refresh))
        except Exception as err:
            return on_invalid(err)

        if verbose:
            print(f"\ngld('{self.name}').optimal_powerflow(refresh={repr(refresh)},verbose={repr(verbose)}{',' if kwargs else ''}{','.join([f'{x}={repr(y)}' for x,y in kwargs.items()])}):",file=sys.stderr)
            print("\nN:",N,sep="\n",file=verbose)
            print("\nG:",G,sep="\n",file=verbose)
            print("\nD:",D,sep="\n",file=verbose)
            print("\nI:",I,sep="\n",file=verbose)
            print("\nF:",F,sep="\n",file=verbose)
            print("\nS:",S,sep="\n",file=verbose)
            print("\nC:",C,sep="\n",file=verbose)
            print("\nTotal D:",np.array(D).sum(),sep="\n",file=verbose)
            print("\nTotal S:",sum(S),sep="\n",file=verbose)
            print("\nTotal C:",sum(C),sep="\n",file=verbose)

        # setup problem
        x = cp.Variable(N)  # nodal voltage angles
        y = cp.Variable(N)  # nodal voltage magnitudes
        g = cp.Variable(N)  # generation real power dispatch
        h = cp.Variable(N)  # generation reactive power dispatch
        c = cp.Variable(N)  # capacitor bank settings
        d = cp.Variable(N)  # demand real power curtailment
        e = cp.Variable(N)  # demand reactive power curtailment

        try:
            cost = P @ cp.abs(g + h * 1j)
            if curtailment_price is None:
                curtailment_price = 100*max(P)
            shed = np.ones(N)*curtailment_price @ cp.abs(d+e*1j) # load shedding 100x maximum generator price
            objective = cp.Minimize(cost + shed)  # minimum cost (generation + demand response)
            constraints = [
                G.real @ x - g + c + D.real - d == 0,  # KCL/KVL real power laws
                G.imag @ y - h - c + D.imag - e == 0,  # KCL/KVL reactive power laws
                x[ref] == 0,  # swing bus voltage angle always 0
                y[ref] == 1,  # swing bus voltage magnitude is always 1
                cp.abs(y - 1) <= voltage_limit,  # limit voltage magnitude to 5% deviation
                cp.abs(I @ x) <= F,  # line flow limits
                g >= 0,  # generation real power limits
                cp.abs(h) <= S.imag,  # generation reactive power limits
                cp.abs(g+h*1j) <= S.real, # generation apparent power limit
                c >= 0, c <= C,  # capacitor bank settings
                d >= 0, cp.abs(d+e*1j) <= cp.abs(D),  # demand curtailment constraint with flexible reactive power
                ]
            problem = cp.Problem(objective, constraints)
            problem.solve(verbose=(verbose!=False),**kwargs)
            self.problem = problem.get_problem_data(solver=problem.solver_stats.solver_name)

        except Exception as err:

            return on_invalid(err)

        if x.value is None:
            return on_fail(problem.status)
        
        status = "inaccurate/approximation" if self.linesplit(angle_limit) else problem.status
        if verbose:
            print(f"Status: {status}\nCost: {problem.value:.2f}",file=verbose)

        puV = self.perunit("V")
        puS = self.perunit("S")
        result = {
                "voltage": np.array([y.value.round(3)*puV,(x.value*57.3).round(2)]).transpose(),
                "angles": self.linevoltage('Va'),
                "generation": np.array((g+h*1j).value).round(3)*puS,
                "capacitors": np.array(c.value).round(3)*puS,
                "flows": cp.abs(I @ x).value.round(3)*puS,
                "cost" : problem.value.round(2),
                "status": status,
                "curtailment":np.array(d.value).round(3)*puS,
            }

        return self.set_result("optimal_sizing",result)

    def optimal_sizing(self,            
            refresh:bool=False,
            verbose:bool|TypeVar('io.TextIOWrapper')=False,
            update_model:bool=False,
            margin:float=0.2,
            gen_cost:float|list|dict=None,
            cap_cost:float|list|dict=None,
            min_power_ratio:float|list|dict=0.1,
            voltage_high:float|list|dict=1.05,
            voltage_low:float|list|dict=0.95,
            steps:float|list|dict=20,
            admittance:float|list|dict=0.1,
            ref:int|str=None,
            angle_limit:float=10.0,
            voltage_limit:float=0.05,
            on_invalid=_problem_invalid,
            on_fail=_solver_failed,
            **kwargs) -> dict:
        """Solve optimal sizing/placement problem

        Arguments:
        * refresh: force recalculation of all values
        * verbose: output solver data and results
        * update_model: update model with new generation and capacitors
        * margin: load capacity margin
        * gen_cost: generation addition cost data
        * cap_cost: capacitor addition cost data
        * min_power_ratio: new generation minimum reactive power ratio relative to real power
        * voltage_high: upper voltage constraint
        * voltage_low: lower voltage constraint
        * steps: number of capacitor steps
        * admittance: capacity admittance per step
        * ref: reference bus id or name
        * angle_limit: voltage angle accuracy limit
        * voltage_limit: voltage magnitude violation limit
        * on_invalid: invalid problem handler
        * on_fail: failed solution handler
        * kwargs: arguments passed to solver

        Returns:
        * dict: results of optimization

        """

        try:
            if not refresh and not update_model:
                return self.get_result("optimal_sizing")
        except:
            pass

        # check model network validity
        if self.graphSpectral()[2] > 1:
            return on_invalid(f"{self.name} cannot optimize more than one network at a time (model has {self.graphSpectral()[2]} networks)")
        if self.graphSpectral()[2] == 0:
            return on_invalid(f"{self.name} cannot optimize on invalid network models (no zero eigenvalues found)")
        
        # setup verbose output
        if verbose is True:
            verbose = sys.stderr

        # extract model data
        try:
            if ref is None:
                ref = self.select({"class":"bus","type":"REF"})
                if len(ref) > 0:
                    if len(ref) > 1:
                        warnings.warn(f"{self.name} multiple reference busses found {ref}, using bus {ref[0]}")
                    ref = ref[list(ref)[0]]["bus_i"]
                else:
                    warnings.warn(f"{self.name} no reference bus found, using bus 0")
                    ref = 0
            elif isinstance(ref,str):
                ref = self.get_bus(ref)

            G = self.graphLaplacian(refresh)
            D = self.demand('actual',refresh)
            I = self.graphIncidence(refresh)
            F = self.lineratings("A",refresh)
            S = self.generation('capacity',refresh)
            C = self.capacitors('installed',refresh)
            N = len(self.nodes(refresh))

            # normalize generation cost argument
            if gen_cost is None:
                gen_cost = np.zeros(N)
            elif isinstance(gen_cost,float) or isinstance(gen_cost,int):
                gen_cost = np.full(N,gen_cost)
            elif isinstance(gen_cost,dict):
                gen_cost = np.array([gen_cost[n] if n in gen_cost else 0 for n in range(N)])
            elif isinstance(gen_cost,list):
                gen_cost = np.array(gen_cost)

            # normalize capacitor cost argument
            if cap_cost is None:
                cap_cost = np.zeros(N)
            elif isinstance(cap_cost,float) or isinstance(cap_cost,int):
                cap_cost = np.full(N,cap_cost)
            elif isinstance(cap_cost,dict):
                cap_cost = np.array([cap_cost[n] if n in cap_cost else 0 for n in range(N)])
            elif isinstance(gen_cost,list):
                cap_cost = np.array(cap_cost)

            # normalize minimum reactive power argument
            if isinstance(min_power_ratio,float):
                min_power_ratio = np.full(N,min_power_ratio)
            elif isinstance(min_power_ratio,dict):
                min_power_ratio = np.array([min_power_ratio[n] if n in min_power_ratio else 0 for n in range(N)])
            elif isinstance(min_power_ratio,list):
                min_power_ratio = np.array(min_power_ratio)

        except Exception as err:

            return on_invalid(err)

        if verbose:
            print(f"\ngld('{self.name}').optimal_sizing(gen_cost={repr(gen_cost)},cap_cost={repr(cap_cost)},refresh={repr(refresh)},update_model={repr(update_model)},margin={repr(margin)},verbose={repr(verbose)}{',' if kwargs else ''}{','.join([f'{x}={repr(y)}' for x,y in kwargs.items()])}):",file=verbose)
            print("\nN:",N,sep="\n",file=verbose)
            print("\nG:",G,sep="\n",file=verbose)
            print("\nD:",D,sep="\n",file=verbose)
            print("\nI:",I,sep="\n",file=verbose)
            print("\nF:",F,sep="\n",file=verbose)
            print("\nS:",S,sep="\n",file=verbose)
            print("\nC:",C,sep="\n",file=verbose)
            print("\nTotal D:",sum(D),sep="\n",file=verbose)
            print("\nTotal S:",sum(S),sep="\n",file=verbose)
            print("\nTotal C:",sum(C),sep="\n",file=verbose)

        # setup problem
        x = cp.Variable(N)  # nodal voltage angles
        y = cp.Variable(N)  # nodal voltage magnitudes
        g = cp.Variable(N)  # generation real power dispatch
        h = cp.Variable(N)  # generation reactive power dispatch
        c = cp.Variable(N)  # capacitor bank settings

        try:

            # construct problem
            puS = self.perunit("S")
            costs = gen_cost.real @ cp.abs(g) + gen_cost.imag @ cp.abs(h) + cap_cost @ cp.abs(c)
            objective = cp.Minimize(costs)  # minimum cost (generation + demand response)
            constraints = [
                g - G.real @ x - c - D.real*(1+margin) == 0,  # KCL/KVL real power laws
                h - G.imag @ y + c - D.imag*(1+margin) == 0,  # KCL/KVL reactive power laws
                x[ref] == 0,  # swing bus voltage angle always 0
                y[ref] == 1,  # swing bus voltage magnitude is always 1
                cp.abs(y - 1) <= voltage_limit,  # limit voltage magnitude to 5% deviation
                cp.abs(I @ x) <= F,  # line flow limits
                g >= 0, # generation must be positive
                c >= 0, # capacitor values must be positive
                ]
            problem = cp.Problem(objective, constraints)
            problem.solve(verbose=(verbose!=False),**kwargs)
            self.problem = problem.get_problem_data(solver=problem.solver_stats.solver_name)

        except Exception as err:

            return on_invalid(err)

        if x.value is None:

            return on_fail(problem.status)

        # update model with new values
        new_gens = [complex(round(max(round(x.real,3)*puS,0),9),round(max(round(x.imag,3)*puS,0),9)) if x.real>0 else 0 for x in (g.value.round(3) + cp.abs(h).value.round(3)*1j - S) ]
        new_caps = [round(max(round(x,3)*puS,0),9) for x in (c.value - C)]
        if update_model:

            if verbose:
                print("\nOSP results:",file=verbose)
                print("-----------",file=verbose)

            # add generators
            if verbose:
                print("\nNew generation:",file=verbose)
                print(f"  Node{' '*(max([len(x) for x in self.find('bus',list)])-4)}    Bus       Pg       Qg      Pmax     Qmax     Qmin  ",file=verbose,)
                print(f"  {'-'*(max([len(x) for x in self.find('bus',list)]))} -------- -------- -------- -------- -------- --------",file=verbose)
            for bus,spec in {self.get_name("bus",n):(n,x) for n,x in enumerate(new_gens) if abs(x)>0}.items():
                gen = f"gen:{len(self.data['objects'])}"
                n = int(self.data['objects'][bus]['bus_i'])-1
                obj = self.add_object("gen",gen,
                    parent=bus,
                    bus=str(self.data['objects'][bus]['bus_i']),
                    Pg = spec[1].real,
                    Qg = spec[1].imag,
                    Pmax=spec[1].real,
                    Qmax=max(spec[1].imag,spec[1].real*min_power_ratio[n]),
                    Qmin=-max(spec[1].imag,spec[1].real*min_power_ratio[n]),
                    status="IN_SERVICE",
                    )
                if verbose:
                    print(' ',' '.join([self.format(self.get_property(gen,x)) for x in ['parent','bus','Pg','Qg','Pmax','Qmax','Qmin']]),file=verbose)
                self.add_object("gencost",f"gencost_{len(self.find('gencost'))}",
                    parent=gen,
                    model="POLYNOMIAL",
                    costs="0.01,100,0", # TODO: where to get this data from (maybe from the lowest cost unit already present if any)
                    )
            
            # add capacitors
            if verbose:
                print("\nNew capacitors:",file=verbose)
                print(f"  Node{' '*(max([len(x) for x in self.find('bus',list)])-4)}   Vhigh    Vlow      Y       Steps    Yc",file=verbose,)
                print(f"  {'-'*(max([len(x) for x in self.find('bus',list)]))} -------- -------- -------- -------- --------",file=verbose)
            for bus,spec in {self.get_name("bus",n):(n,x) for n,x in enumerate(new_caps) if abs(x)>0}.items():
                shunt = f"shunt:{len(self.data['objects'])}"
                self.add_object("shunt",shunt,
                    parent=bus,
                    voltage_high=voltage_high,
                    voltage_low=voltage_low,
                    admittance=spec[1],
                    steps_1=steps,
                    admittance_1=admittance,
                    )
                if verbose:
                    print(' ',' '.join([self.format(self.get_property(shunt,x)) for x in ['parent','voltage_high','voltage_low','admittance','steps_1','admittance_1']]),file=verbose)

        status = "inaccurate/approximation" if self.linesplit(angle_limit) else problem.status
        if verbose:
            print(f"Status: {status}\nCost: {problem.value:.2f}",file=verbose)

        # generate result data
        puV = self.perunit("V")
        result = {
                "voltage": np.array([y.value.round(3)*puV,(x.value*57.3).round(2)]).transpose(),
                "angles": self.linevoltage('Va'),
                "generation": np.array((g+h*1j).value).round(3)*puS,
                "capacitors": np.array(c.value).round(3)*puS,
                "flows": cp.abs(I @ x).value.round(3)*puS,
                "cost" : problem.value.round(2),
                "status": status,
                "additions": {
                    "generation": {n:x for n,x in enumerate(new_gens) if abs(x)>0},
                    "capacitors": {n:x for n,x in enumerate(new_caps) if abs(x)>0},
                }
            }

        return self.set_result("optimal_sizing",result)

    #
    # PyPOWER
    #

    pypower = {
        "bus": ["bus_i","type","Pd","Qd","Gs","Bs","area","Vm","Va","baseKV","zone","Vmax","Vmin"],
        "branch": ["fbus","tbus","r","x","b","rateA","rateB","rateC","ratio","angle","status","angmin","angmax"],
        "gen": ["bus","Pg","Qg","Qmax","Qmin","Vg","mBase","status","Pmax","Pmin","Pc1","Pc2","Qc1min","Qc1max","Qc2min","Qc2max","ramp_agc","ramp_10","ramp_30","ramp_q","apf"],
        "gencost": ["model","startup","shutdown"],
        }

    def as_case(self) -> dict:

        case = {
            "version": "2",
            "baseMVA": self.globals("pypower::baseMVA"),
            }
        for name,fields in self.pypower.items():
            case[name] = [[self.get_property(x,y,astype=float) for y in fields] for x in self.find(name)]
        costs = [[float(y) for y in self.get_property(x,"costs").split(",")] for x in self.find("gencost")]
        for n,cost in enumerate(costs):
            case["gencost"][n].extend([len(cost)]+cost)
        if len(case["gencost"]) == 0:
            del case["gencost"]

        for array in self.pypower:
            if array in case:
                case[array] = np.array(case[array])

        return case

    def savecase(self,file):
        """Save pypower case data"""
        with open(file,"w") as fh:
            print(f"""from numpy import array
def {os.path.splitext(os.path.basename(self.name))[0]}():
    ppc = {{}}""",file=fh)
            for key,value in self.as_case().items():
                if hasattr(value,"tolist"):
                    print(f"""    ppc["{key}"] = array([""",file=fh)
                    print(f"""        # {" ".join([f"{x:9.9s}" for x in self.pypower[key]])}""",file=fh)
                    for row in value.tolist():
                        print(f"""        [{", ".join([f"{x:8.4g}" for x in row])}],""",file=fh)
                    print(f"""    ])""",file=fh)
                else:
                    print(f"""    ppc["{key}"] = {value}""",file=fh)

    def runpf(self,casedata=None,**kwargs) -> dict:
        """Run pypower powerflow solver"""
        return runpf(self.as_case() if casedata is None else casedata,ppoption(**kwargs))

    def runopf(self,casedata=None,**kwargs) -> dict:
        """Run pypower optimal powerflow solver"""
        return runopf(self.as_case() if casedata is None else casedata,ppoption(**kwargs))

    def mermaid(self,
        orientation:str="vertical",
        label=None,
        overvolt:float=1.05,
        undervolt:float=0.95,
        highflow:float=1.0,
        showbusdata:Union[bool,list]=False,
        showarea:str=None,
        showpopup:Union[bool,list]=False,
        showloads:bool=True,
        showgens:bool=True,
        ) -> str:
        """Generate network diagram in Mermaid

        Arguments:
        * orientation: horizontal or vertical graph orientation
        * label: property to use as label
        * overvolt: voltage limit for red fill
        * undervolt: voltage limit for blue fill
        * highflow: current limit for heavy line
        * showbusdata: enable display of bus data (or list of properties to display)
        * showarea: limit display to area
        * showpopup: include popup data
        * showloads: include loads
        * showgens: include generators

        Returns:
        * str: Mermaid diagram string
        """
        orientations = {"vertical":"TB","horizontal":"LR"}
        diagram = [f"""graph {orientations[orientation]}
    classDef black fill:#000,stroke:#000;
    classDef white fill:#fff,stroke:#000;
    classDef red fill:#f64,stroke:#000;
    classDef green fill:#0f0,stroke:#000;
    classDef blue fill:#6cf,stroke:#000;
"""]
        if showbusdata == True:
            showbusdata = ["id","type","area","Vm","Va","zone"]
        elif showbusdata == False:
            showbusdata = []
        elif not isinstance(showbusdata,list) :
            raise ValueError("showbusdata is not a list or bool")
        def _node(bus,spec):
            node = spec["bus_i"]
            name = spec[label] if label else bus
            Vm = self.get_property(bus,"Vm")
            Pd = self.get_property(bus,"Pd")
            Qd = self.get_property(bus,"Qd")
            gens = self.select({"class":"gen","bus":node})
            loads = self.select({"class":"load","parent":bus})
            Pg = sum([self.get_property(x,"Pg") for x in gens])
            Qg = sum([self.get_property(x,"Qg") for x in gens])
            shape = "rect" if showbusdata else "fork"
            busdata = "".join([f"<div><b><u>{name}</u></b></div>"]+[f"<div><b>{x}</b>: {y}</div>" for x,y in spec.items() if x in showbusdata])
            busdata = "".join([f"<table><caption><b>{name}</b><hr/></caption>"]+[f"<tr><th align=left>{x}</th><td align=center>:</td><td align=right>{y.split()[0]}</td><td align=right>{y.split(' ',1)[1] if ' ' in y else ''}</td></tr>" for x,y in spec.items() if x in showbusdata]) + "</table>"
            result = [f"""    {node}@{{shape: {shape}, label: "{busdata}"}}"""]

            if not undervolt is None and Vm < undervolt:
                color = "blue"
            elif not overvolt is None and Vm > overvolt:
                color = "red"
            elif showbusdata:
                color = "white"
            else:
                color = "black"
            result.append(f"""    class {node} {color}""")

            if abs(complex(Pg,Qg)) > 0 and showgens:
                result.append(f"""    G{node}@{{shape: circle, label: "<div>{name}</div>"}} --{Pg:.1f}{Qg:+.1f}j MVA--> {node}""")
                result.append(f"""    class G{node} white""")
            if ( abs(complex(Pd,Qd)) > 0 or len(loads) > 0 ) and showloads:
                result.append(f"""    {node} --{Pd:.1f}{Qd:+.1f}j MVA--> L{node}@{{shape: tri, label: "<div>{name}</div>"}}""")
                result.append(f"""    class L{node} white""")

            return "\n".join(result)

        if showarea is None:
            busses = self.find("bus")
        else:
            busses = self.select({"class":"bus","area":showarea})
        for bus,spec in busses.items():

            diagram.append(_node(bus,spec))

        baseMVA = self.perunit('S')
        def _line(line,spec):
            fbus = spec["tbus"]
            tbus = spec["fbus"]
            names = self.get_name('bus',[int(fbus)-1,int(tbus)-1])
            baseKV = self.get_property(names[0],"baseKV")
            baseZ = baseKV**2/baseMVA
            voltages = self.get_property(names,["Vm","Va"])
            def cpdiff(x): # complex polar difference
                # print(f"{x=}")
                a0,a1 = [y.imag*math.pi/180 for y in x]
                v0 = x[0].real*(math.cos(a0)+math.sin(a0)*1j)
                v1 = x[1].real*(math.cos(a1)+math.sin(a1)*1j)
                dv = v1 - v0
                # print(f"{dv=}")
                return v1-v0
            voltage = cpdiff([complex(*voltages[x]) for x in names])
            impedance = complex(float(spec['r'].split()[0]),float(spec['x'].split()[0]))
            current = voltage / ( impedance if abs(impedance) > 0 else 1e-6 )
            # print(line,[fbus,tbus],names,baseKV,baseMVA,baseZ,voltages,voltage,impedance,current)
            # current = self.get_property(line,"current")
            power = voltage * current.conjugate()
            # print(f"{line=}: {baseZ=}, {baseKV=}, {baseMVA=}")
            # print(f"  {voltage=:.4f}")
            # print(f"  {impedance=:.4f}")
            # print(f"  {current=:.4f}")
            # print(f"  {power=:.4f}")
            reverse = ( current.real < 0 )
            current = abs(current/1000)
            linetype = "--" if not highflow is None and current < highflow else "=="
            fbus = spec["tbus" if reverse else "fbus"]
            tbus = spec["fbus" if reverse else "tbus"]
            return f"""    {fbus} {linetype}{current:.2f} kA<br>{power*baseMVA:.1f} MVA{linetype}> {tbus}"""

        for line,spec in self.find("branch").items():
            if self.get_name("bus",int(spec["fbus"])-1) in busses or self.get_name("bus",int(spec["tbus"])-1) in busses:
                diagram.append(_line(line,spec))

        if showpopup == True or isinstance(showpopup,list):
            for bus,spec in busses.items():
                popup = f""" "{"<br>".join([f"<b>{x}</b>: {y}" for x,y in spec.items() if showpopup ==True or x in showpopup])}" """.strip()
                diagram.append(f"""    click {spec["bus_i"]} callback {popup}\n""")
                diagram.append(f"""    click {spec["bus_i"]} call callback() {popup}""")

        return "\n".join(diagram)

if __name__ == "__main__":

    if not os.path.exists("example.json"):
        print("TEST: example.json not found, testing not done",file=sys.stderr)
        quit()

    test = Model("example.json")

    try:
        test.run("--version")
        runtime=True
    except RuntimeError as err:
        print("\nWARNING: GridLAB-D not installed, skipping runtime tests",file=sys.stderr)
        runtime = False

    tested = 0
    failed = 0

    def testEq(a,b,msg):
        import inspect
        global tested
        tested += 1
        if a != b:
            caller = inspect.getframeinfo(inspect.stack()[1][0])
            print(f"TEST [{os.path.basename(caller.filename)}@{caller.lineno}]: {msg}: {repr(a)} != {repr(b)}",file=sys.stderr,flush=True)
            global failed
            failed += 1
    def testIn(a,b,msg):
        import inspect
        global tested
        tested += 1
        if not a in b:
            caller = inspect.getframeinfo(inspect.stack()[1][0])
            print(f"TEST [{os.path.basename(caller.filename)}@{caller.lineno}]: {msg}: {repr(a)} != {repr(b)}",file=sys.stderr,flush=True)
            global failed
            failed += 1
    def testException(a,exc,msg):
        try:
            a()
            testEq(None,exc.__name__,msg)
        except:
            e_type,e_value,e_trace = sys.exc_info()
            testEq(e_type.__name__,exc.__name__,msg)


    if runtime:
        testEq(test.run(exception=False),(0, [''], ['']),"initial run test failed")

    bus_3 = test.get_object("bus_3")

    # accessor tests
    print("TEST: testing accessors",file=sys.stderr,flush=True)
    testEq(test.get_property("bus_0",'id'),2,"get header failed")
    testEq(bus_3["bus_i"],'4',"get object failed")
    testException(lambda:test.add_object("bus","bus_3",**bus_3)["bus_i"],ValueError,"add object succeeded")
    testException(lambda:test.add_object("bus","bus_4",id="0")["bus_i"],ValueError,"add object succeeded")
    testEq(test.del_object("bus_3"),bus_3,"del object failed")
    testEq(test.add_object("bus","bus_3",**bus_3)["bus_i"],bus_3["bus_i"],"add object failed")
    testException(lambda:test.add_object("bus","bus_0"),ValueError,"add object failed")
    testException(lambda:test.add_object("transformer","test"),ValueError,"add object failed")
    testEq(test.add_object("geodata","test",scale=0.1),{'class': 'geodata', 'id': "13", 'scale': '0.1 pu'},"add object failed")
    testEq(test.mod_object("test",scale=1.0),{'class': 'geodata', 'id': "13", 'scale': '1.0 pu'},"mod object failed")
    testEq(test.del_object("test"),{'class': 'geodata', 'id': "13", 'scale': '1.0 pu'},"add object failed")

    # content tests
    print("TEST: testing model contents",file=sys.stderr,flush=True)
    testEq("pypower" in test.modules(),True,"module failed")
    testEq(test.validate(["pypower"]),None, "validate failed")
    testEq("version" in test.globals(list),True,"globals list failed")
    testEq(test.globals(dict)["country"],"US", "globals dict failed" )
    testEq(test.globals("country"),"US", "globals get failed")
    testEq(test.find("bus",list),['bus_0', 'bus_1', 'bus_2', 'bus_3'], "find list failed")
    testEq([y['bus_i'] for y in test.find("bus",dict).values()],['1','2','3','4'], "find dict failed")
    testEq(list(test.select({"class":"bus","type":"REF"})),['bus_0'],"select failed")
    testEq(test.get_name('bus') , ['bus_0', 'bus_1', 'bus_2', 'bus_3'], "get bus name failed")
    testEq(test.get_name('bus',0) , 'bus_0', "get bus name failed")
    testEq(test.get_name('bus',[1,2]) , ['bus_1', 'bus_2'], "get bus name failed")
    testEq(test.get_name('branch') , ['branch:6', 'branch:7', 'branch:8'], "get branch failed")
    testEq(test.get_name('branch',0) , 'branch:6', "get branch failed")
    testEq(test.get_name('branch',[1,2]) , ['branch:7', 'branch:8'], "get branch failed")
    testEq(test.get_bus("gen_0") , "bus_0", "get bus failed")
    testEq(test.get_bus(["gen_0"]) , ["bus_0"], "get bus failed")
    testEq(test.get_property("bus_0","Pd"),0.0, "property float failed")
    testEq(test.get_property("bus_0","S"),0j, "property complex failed")
    testEq(test.perunit("S"),100, "perunit power failed")
    testEq(test.perunit("V"),[12.5, 12.5, 12.5, 12.5], "perunit voltage failed")
    testEq(test.perunit("Z"),[1.5625, 1.5625, 1.5625], "perunit impedance failed")
    testEq(test.graphLaplacian().shape,(4,4), "graph Laplacian failed")
    testEq(test.graphIncidence().shape,(3,4), "graph incidence failed")
    testEq(test.demand().tolist(),[0j,0j,0.1+0.01j,0.1+0.01j], "demand failed")
    testEq(list(test.generators().keys()) , ['gen_0'], "generators failed")
    testEq(test.generation().tolist() , [(0.1+0.05j), 0j, 0j, 0j], "generation failed")
    testEq(list(test.costs().keys()) , ['gencost:1'], "costs failed")
    testEq(test.prices().tolist() , [0,0,0,0], "prices failed")
    testEq(test.lineratings().tolist() , [0.25,0.15,0.15], "line ratings failed")
    testEq(test.capacitors().tolist() , [0,0,0,0], "capacitors failed")
    testEq(test.mermaid().split("\n")[0],"graph TB","mermaid failed")

    # optimization tests
    print("TEST: testing optimizations",file=sys.stderr,flush=True)
    testEq(test.optimal_powerflow()["curtailment"].round(1).tolist(),[0.0, 0.0, 6.8, 6.8],"optimal powerflow failed")
    testEq(test.optimal_sizing(refresh=True,gen_cost=np.array([100,500,1000,1000])+1000j,cap_cost={0:1000,1:500})["generation"].round(1).tolist() , [(26.4+0j), 0j, 0j, 0j], "optimal sizing failed")
    testEq(test.optimal_sizing(refresh=True,gen_cost=np.array([100,500,1000,1000])+1000j,cap_cost={0:1000,1:500})["capacitors"].round(1).tolist() , [0,0,1.2,1.2], "optimal sizing failed")
    testEq(test.optimal_sizing(refresh=True,gen_cost=np.array([100,500,1000,1000])+1000j,cap_cost={0:1000,1:500},update_model=True)["additions"] , {'generation': {0: (16.4+0j)}, 'capacitors': {2: 1.2, 3: 1.2}} , "optimal sizing failed")
    testEq(test.optimal_powerflow(refresh=True)["curtailment"].tolist(),[0,0,0,0],"optimal powerflow failed")
    if runtime:
        test.data["globals"]["savefile"] = ""
        test.save("test_out.json",indent=4)
        rc,out,err = test.run("test_out.json",exception=False)
        testEq(out,[''],'run test failed')

    # case tests
    print("TEST: testing pypower cases",file=sys.stderr,flush=True)
    for file in sorted(os.listdir("test")):
        if file.startswith("case") and file.endswith(".json"):

            test = Model(os.path.join("test",file))

            # pypower PF test
            try:
                testEq(test.runpf(OUT_ALL=0,VERBOSE=0)[1],1,f"{file} runpf failed")
            except Exception as err:
                print(f"ERROR: {file} runpf raised exception {err}",file=sys.stderr)
                test.savecase("test/"+file.replace(".json","_runpf_failed.py"))
                failed += 1

            # pypower OPF test
            try:
                if test.find("gencost"):
                    testEq(test.runopf(OUT_ALL=0,VERBOSE=0)["success"],True,f"{file} runopf failed")
            except Exception as err:
                print(f"ERROR: {file} runopf raised exception {err}",file=sys.stderr)
                test.savecase("test/"+file.replace(".json","_runopf_failed.py"))
                failed += 1

            # enhanced OPF test
            if not test.optimal_powerflow(on_fail=lambda x:print(f"\nTEST: {file} initial OPF is {x}",file=sys.stderr)):
                test.optimal_powerflow(verbose=True,on_fail=lambda x: print(test.problem,file=sys.stderr))
                failed += 1
            tested += 1
            split = test.linesplit()
            if split:
                print(file,test.linesplit(update_model=True),file=sys.stderr)

            # OSP test
            testIn(test.optimal_sizing(refresh=True,angle_limit=10,update_model=True)["status"],["optimal","inaccurate/approximation"],f"{file} sizing failed")

            # OSP/OPF test
            testEq(test.optimal_powerflow(refresh=True)["curtailment"].tolist(),np.zeros(len(test.find("bus"))).tolist(),f"{file} final OPF failed")

    print("TEST: completed",tested,"tests",file=sys.stderr,flush=True)
    if failed:
        print("ERROR:",failed,"test failed",file=sys.stderr)
    else:
        print("TEST: no errors",file=sys.stderr)
