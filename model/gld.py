"""GridLAB-D optimal powerflow/sizing/placement

Example:

The following example loads the 4-bus model and attempts an OPF. However,
there is insufficient generation to avoid curtailment. Then it runs
the optimal sizing/placement problem and updates the model with the result.
Then the OPF runs without curtailment and the simulation is run with the new model.

>>> import gld
>>> test = Model("4bus.json")
>>> test.optimal_powerflow()["curtailment"]
>>> test.optimal_sizing(gen_cost=np.array([100,500,1000,1000])+1000j,
                        cap_cost={0:1000,1:500},
                        update_model=True)
>>> test.optimal_powerflow(refresh=True)["curtailment"]
>>> test.run("4bus_test.json")
"""

import sys
import os
import json
import io
import subprocess
import numpy as np
import numpy.linalg as la
import cvxpy as cp
from typing import Union, Any, TypeVar

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

    def property(self,object:str,property:str) -> Any:
        """Get an object property and convert to Python type

        Arguments:
        * object: name of object
        * property: name of property

        Returns:
        * varies: value of object property
        """
        object_data = self.data["objects"][object]
        if property in self.data["header"]:
            return object_data[property]
        ptype = self.data["classes"][object_data["class"]][property]["type"]
        if ptype in dir(self):
            return getattr(self,ptype)(object_data[property])
        return object_data[property]

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
        """Find objects

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
        if self.property(name,"class") == "bus":
            return name
        parent = self.property(name,"parent")
        return self.get_bus(parent)

    def get_branch(self,kind:str,id:int|list[int]=None):
        return None

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
            self.results["perunit"+kind] = [self.property(x,"baseKV") for x in self.find("bus")]
        elif kind == 'Z':
            names = {y["bus_i"]:x for x,y in self.find("bus").items()}
            self.results["perunit"+kind] = [self.property(names[x["fbus"]],"baseKV")**2/self.globals("pypower::baseMVA") for x in self.find("branch",dict).values()]
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
            return self.result["costs"]
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
        self.results["impedance"] = [complex(self.property(x,"r"),self.property(x,"x")) for x in self.lines(refresh)]
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
            G[l[0], l[1]] = G[l[1], l[0]] = 1 / R[n]
        self.results["graphLaplacian"] = np.diag(sum(G)) - G # graph Laplacian
        return self.results["graphLaplacian"]

    def graphIncidence(self,refresh:bool=False) -> np.array:
        """Get network indicidence matrix

        Arguments:
        * refresh: force recalculation of previous results

        Returns:
        * np.array: incidence matrix
        """
        if "graphIncidence" in self.results and not refresh:
            return self.results["graphIncidence"]
        self.assert_module("pypower")
        lines = self.find("branch")
        N = len(self.find("bus",list))
        L = len(lines)
        B = np.array([[int(x["fbus"])-1,int(x["tbus"])-1] for x in lines.values()])
        R = [self.property(x,"r")+self.property(x,"x")*1j for x in lines]
        I = np.zeros((L, N))  # link-node incidence matrix
        for n, l in enumerate(B):
            I[n][l[0]] = -R[n].real
            I[n][l[1]] = R[n].real
        self.results["graphIncidence"] = I
        return self.results["graphIncidence"]

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
            self.results[f"demand.{kind}" ] = np.array([complex(self.property(x,"Pd"),self.property(x,"Qd")) for x in self.nodes(refresh)]) / self.perunit("S")
        elif kind == "peak":
            self.results[f"demand.{kind}" ] = np.array([complex(self.property(x,"Pd"),self.property(x,"Qd")) for x in self.nodes(refresh)]) / self.perunit("S")
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
            gen = [(self.property(x,"bus"),complex(self.property(x,"Pmax")/puS,self.property(x,"Qmax")/puS)) for x in self.generators(refresh)]
        elif kind == 'actual':
            gen = [(self.property(x,"bus"),complex(self.property(x,"Pg")/puS,self.property(x,"Qg")/puS)) for x in self.generators(refresh)]
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
        costs = {self.property(y["parent"],"bus"):float(y["costs"].split(",")[1]) for x,y in self.costs(refresh).items()}
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
        result = {y["parent"]:{'setting':self.property(x,"admittance"),'capacity':self.property(x,"admittance_1")*self.property(x,"steps_1")} for x,y in self.find("shunt").items()}
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
            cap = [(self.property(x,"bus_i"),shunts[x]["capacity"]/puS if x in shunts else 0.0) for x in self.nodes(refresh)]
        elif kind == 'setting':
            cap = [(self.property(x,"bus_i"),shunts[x]["setting"]/puS if x in shunts else 0.0) for x in self.nodes(refresh)]
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
        self.results[f"lineratings.{rating}"] = np.array([self.property(x,f"rate{rating}") for x in self.lines()])/self.perunit('S',refresh) 
        return self.results[f"lineratings.{rating}"]

    #
    # Optimizations
    #

    def optimal_powerflow(self,
        refresh:bool=False,
        with_solver_data:bool=False,
        verbose:bool=False,
        **kwargs) -> dict:
        """Compute optimal powerflow

        Arguments:
        * refresh: force recalculation of previous result
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
            raise RuntimeError(f"cannot solve OPF on more than one network at a time (model has {self.graphSpectral()[2]} networks)")
        if self.graphSpectral()[2] == 0:
            raise RuntimeError(f"cannot solve OPF on invalid network modles (no zero eigenvalues found)")

        # extract data from model        
        P = self.prices(refresh)
        G = self.graphLaplacian(refresh)
        D = self.demand('actual',refresh)
        I = self.graphIncidence(refresh)
        F = self.lineratings("A",refresh)
        S = self.generation('capacity',refresh)
        C = self.capacitors('installed',refresh)
        N = len(self.nodes(refresh))

        if verbose:
            print(f"\ngld('{self.name}').optimal_powerflow(refresh={repr(refresh)},with_solver_data={repr(with_solver_data)},verbose={repr(verbose)}{',' if kwargs else ''}{','.join([f'{x}={repr(y)}' for x,y in kwargs.items()])}):",file=sys.stderr)
            print("\nN:",N,sep="\n",file=sys.stderr)
            print("\nG:",G,sep="\n",file=sys.stderr)
            print("\nD:",D,sep="\n",file=sys.stderr)
            print("\nI:",I,sep="\n",file=sys.stderr)
            print("\nF:",F,sep="\n",file=sys.stderr)
            print("\nS:",S,sep="\n",file=sys.stderr)
            print("\nC:",C,sep="\n",file=sys.stderr)

        # setup problem
        x = cp.Variable(N)  # nodal voltage angles
        y = cp.Variable(N)  # nodal voltage magnitudes
        g = cp.Variable(N)  # generation real power dispatch
        h = cp.Variable(N)  # generation reactive power dispatch
        c = cp.Variable(N)  # capacitor bank settings
        d = cp.Variable(N)  # demand curtailment

        cost = P @ cp.abs(g + h * 1j)
        shed = np.ones(N)*100*max(P) @ d # load shedding 100x maximum generator price
        objective = cp.Minimize(cost + shed)  # minimum cost (generation + demand response)
        constraints = [
            G.real @ x - g + c + D.real - d == 0,  # KCL/KVL real power laws
            G.imag @ y - h - c + D.imag - d @ D.imag / D.real == 0,  # KCL/KVL reactive power laws
            x[0] == 0,  # swing bus voltage angle always 0
            y[0] == 1,  # swing bus voltage magnitude is always 1
            cp.abs(y - 1) <= 0.05,  # limit voltage magnitude to 5% deviation
            cp.abs(I @ x) <= F,  # line flow limits
            g >= 0, g <= S.real,  # generation real power limits
            cp.abs(h) <= S.imag,  # generation reactive power limits
            c >= 0, c <= C,  # capacitor bank settings
            d >= 0, d <= D.real,  # demand curtailment
            ]
        problem = cp.Problem(objective, constraints)
        problem.solve(**kwargs)

        if x.value is None:
            raise RuntimeError(problem.status)
        
        puV = self.perunit("V")
        puS = self.perunit("S")
        result = {
                "voltage": np.array([y.value.round(3)*puV,x.value.round(3)*57.3]).transpose(),
                "generation": np.array((g+h*1j).value).round(3)*puS,
                "capacitors": np.array(c.value).round(3)*puS,
                "flows": cp.abs(I @ x).value.round(3)*puS,
                "cost" : problem.value.round(2),
                "status": problem.status,
                "curtailment":np.array(d.value).round(3)*puS,
            }
        if with_solver_data:
            result["problem"] = problem.get_problem_data(problem.solver_stats.solver_name)

        return self.set_result("optimal_sizing",result)

    def optimal_sizing(self,            
            refresh:bool=False,
            verbose:bool|TypeVar('io.TextIOWrapper')=False,
            with_solver_data:bool=False,
            update_model:bool=False,
            margin:float=0.2,
            gen_cost:float|list|dict=None,
            cap_cost:float|list|dict=None,
            min_power_ratio:float|list|dict=0.1,
            voltage_high:float|list|dict=1.05,
            voltage_low:float|list|dict=0.95,
            steps:float|list|dict=20,
            admittance:float|list|dict=0.1,
            **kwargs) -> dict:
        """Solve optimal sizing/placement problem

        Arguments:
        * refresh: force recalculation of all values
        * verbose: output solver data and results
        * with_solver_data: include solver data
        * update_model: update model with new generation and capacitors
        * margin: load safety margin (default is +0.2)
        * gen_cost: generation addition cost data
        * cap_cost: capacitor addition cost data
        * min_power_ratio: new generation minimum reactive power ratio relative to real power
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
            raise RuntimeError(f"cannot optimize more than one network at a time (model has {self.graphSpectral()[2]} networks)")
        if self.graphSpectral()[2] == 0:
            raise RuntimeError(f"cannot optimize on invalid network models (no zero eigenvalues found)")
        
        # setup verbose output
        if verbose is True:
            verbose = sys.stderr

        # extract model data
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

        if verbose:
            print(f"\ngld('{self.name}').optimal_sizing(gen_cost={repr(gen_cost)},cap_cost={repr(cap_cost)},refresh={repr(refresh)},with_solver_data={repr(with_solver_data)},update_model={repr(update_model)},margin={repr(margin)},verbose={repr(verbose)}{',' if kwargs else ''}{','.join([f'{x}={repr(y)}' for x,y in kwargs.items()])}):",file=verbose)
            print("\nN:",N,sep="\n",file=verbose)
            print("\nG:",G,sep="\n",file=verbose)
            print("\nD:",D,sep="\n",file=verbose)
            print("\nI:",I,sep="\n",file=verbose)
            print("\nF:",F,sep="\n",file=verbose)
            print("\nS:",S,sep="\n",file=verbose)
            print("\nC:",C,sep="\n",file=verbose)

        # setup problem
        x = cp.Variable(N)  # nodal voltage angles
        y = cp.Variable(N)  # nodal voltage magnitudes
        g = cp.Variable(N)  # generation real power dispatch
        h = cp.Variable(N)  # generation reactive power dispatch
        c = cp.Variable(N)  # capacitor bank settings

        # construct problem
        puS = self.perunit("S")
        costs = gen_cost.real @ cp.abs(g) + gen_cost.imag @ cp.abs(h) + cap_cost @ cp.abs(c)
        objective = cp.Minimize(costs/puS)  # minimum cost (generation + demand response)
        constraints = [
            g - G.real @ x - c - D.real*(1+margin) == 0,  # KCL/KVL real power laws
            h - G.imag @ y + c - D.imag*(1+margin) == 0,  # KCL/KVL reactive power laws
            x[0] == 0,  # swing bus voltage angle always 0
            y[0] == 1,  # swing bus voltage magnitude is always 1
            cp.abs(y - 1) <= 0.05,  # limit voltage magnitude to 5% deviation
            cp.abs(I @ x) <= F,  # line flow limits
            g >= 0, # generation must be positive
            c >= 0, # capacitor values must be positive
            ]
        problem = cp.Problem(objective, constraints)
        problem.solve(**kwargs)

        if x.value is None:
            raise RuntimeError(problem.status)

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
                gen = f"gen_{len(self.find('gen'))}"
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
                    print(' ',' '.join([self.format(self.property(gen,x)) for x in ['parent','bus','Pg','Qg','Pmax','Qmax','Qmin']]),file=verbose)
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
                shunt = f"shunt_{len(self.find('shunt'))}"
                self.add_object("shunt",shunt,
                    parent=bus,
                    voltage_high=voltage_high,
                    voltage_low=voltage_low,
                    admittance=spec[1],
                    steps_1=steps,
                    admittance_1=admittance,
                    )
                if verbose:
                    print(' ',' '.join([self.format(self.property(shunt,x)) for x in ['parent','voltage_high','voltage_low','admittance','steps_1','admittance_1']]),file=verbose)

        if verbose:
            print(f"Cost: {problem.value:.2f}",file=verbose)

        # generate result data
        puV = self.perunit("V")
        result = {
                "voltage": np.array([y.value.round(3)*puV,(x.value*57.3).round(2)]).transpose(),
                "generation": np.array((g+h*1j).value).round(3)*puS,
                "capacitors": np.array(c.value).round(3)*puS,
                "flows": cp.abs(I @ x).value.round(3)*puS,
                "cost" : problem.value.round(2),
                "status": problem.status,
                "additions": {
                    "generation": {n:x for n,x in enumerate(new_gens) if abs(x)>0},
                    "capacitors": {n:x for n,x in enumerate(new_caps) if abs(x)>0},
                }
            }
        if with_solver_data:
            result["problem"] = problem.get_problem_data(problem.solver_stats.solver_name)

        return self.set_result("optimal_sizing",result)

if __name__ == "__main__":

    if not os.path.exists("4bus.json"):
        print("TEST: 4bus.json not found, testing not done",file=sys.stderr)
        quit()

    test = Model("4bus.json")

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
            print(f"\nTEST [{os.path.basename(caller.filename)}@{caller.lineno}]: {msg}: {repr(a)} != {repr(b)}",file=sys.stderr,flush=True)
            global failed
            print("TEST: continuing tests",end="",file=sys.stderr,flush=True)
            failed += 1
        else:
            print(".",end="",flush=True,file=sys.stderr)
    def testException(a,exc,msg):
        try:
            a()
            testEq(None,exc.__name__,msg)
        except:
            e_type,e_value,e_trace = sys.exc_info()
            testEq(e_type.__name__,exc.__name__,msg)

    print("TEST: starting tests",end="",file=sys.stderr,flush=True)

    if runtime:
        testEq(test.run(),"","initial run test failed")

    bus_3 = test.get_object("bus_3")
    testEq(bus_3["bus_i"],'4',"get object failed")
    testException(lambda:test.add_object("bus","bus_3",**bus_3)["bus_i"],ValueError,"add object succeeded")
    testException(lambda:test.add_object("bus","bus_4",id="0")["bus_i"],ValueError,"add object succeeded")
    testEq(test.del_object("bus_3"),bus_3,"del object failed")
    testEq(test.add_object("bus","bus_3",**bus_3)["bus_i"],bus_3["bus_i"],"add object failed")
    testException(lambda:test.add_object("bus","bus_0"),ValueError,"add object failed")
    testException(lambda:test.add_object("transformer","test"),ValueError,"add object failed")
    testEq(test.add_object("geodata","test",scale=0.1),{'class': 'geodata', 'id': "10", 'scale': '0.1 pu'},"add object failed")
    testEq(test.mod_object("test",scale=1.0),{'class': 'geodata', 'id': "10", 'scale': '1.0 pu'},"mod object failed")
    testEq(test.del_object("test"),{'class': 'geodata', 'id': "10", 'scale': '1.0 pu'},"add object failed")

    testEq(os.path.exists("4bus.json"),True,"test model 4bus.json not found")
    testEq("pypower" in test.modules(),True,"module failed")
    testEq(test.validate(["pypower"]),None, "validate failed")
    testEq("version" in test.globals(list),True,"globals list failed")
    testEq(test.globals(dict)["country"],"US", "globals dict failed" )
    testEq(test.globals("country"),"US", "globals get failed")
    testEq(test.find("bus",list),['bus_0', 'bus_1', 'bus_2', 'bus_3'], "find list failed")
    testEq([y['bus_i'] for y in test.find("bus",dict).values()],['1','2','3','4'], "find dict failed")
    testEq(test.get_name('bus') , ['bus_0', 'bus_1', 'bus_2', 'bus_3'], "get bus name failed")
    testEq(test.get_name('bus',0) , 'bus_0', "get bus name failed")
    testEq(test.get_name('bus',[1,2]) , ['bus_1', 'bus_2'], "get bus name failed")
    testEq(test.get_name('branch') , ['branch:4', 'branch:5', 'branch:6'], "get branch failed")
    testEq(test.get_name('branch',0) , 'branch:4', "get branch failed")
    testEq(test.get_name('branch',[1,2]) , ['branch:5', 'branch:6'], "get branch failed")
    testEq(test.get_bus("gen:7") , "bus_0", "get bus failed")
    testEq(test.get_bus(["gen:7"]) , ["bus_0"], "get bus failed")
    testEq(test.get_bus("shunt:9") , "bus_3", "get bus failed")
    testEq(test.get_bus(["shunt:9"]) , ["bus_3"], "get bus failed")
    testEq(test.property("bus_0","Pd"),0.0, "property float failed")
    testEq(test.property("bus_0","S"),0j, "property complex failed")
    testEq(test.perunit("S"),100, "perunit power failed")
    testEq(test.perunit("V"),[12.5, 12.5, 12.5, 12.5], "perunit voltage failed")
    testEq(test.perunit("Z"),[1.5625, 1.5625, 1.5625], "perunit impedance failed")
    testEq(test.graphLaplacian().shape,(4,4), "graph Laplacian failed")
    testEq(test.graphIncidence().shape,(3,4), "graph incidence failed")
    testEq(test.demand().tolist(),[0j,0j,0.1+0.01j,0.1+0.01j], "demand failed")
    testEq(list(test.generators().keys()) , ['gen:7'], "generators failed")
    testEq(test.generation().tolist() , [complex(0.1,0.01),0j,-0j,0j], "generation failed")
    testEq(list(test.costs().keys()) , ['gencost:8'], "costs failed")
    testEq(test.prices().tolist() , [100,0,0,0], "prices failed")
    testEq(test.lineratings().tolist() , [0.25,0.15,0.15], "line ratings failed")
    testEq(test.capacitors().tolist() , [0,0,0,0.1], "capacitors failed")

    testEq(test.optimal_powerflow()["curtailment"].tolist(),[0,0,5,5],"optimal powerflow failed")
    testEq(test.optimal_sizing(refresh=True,gen_cost=np.array([100,500,1000,1000])+1000j,cap_cost={0:1000,1:500})["generation"].round(1).tolist() , [24.8+0j, 1.6+0j, 0j, 0j], "optimal sizing failed")
    testEq(test.optimal_sizing(refresh=True,gen_cost=np.array([100,500,1000,1000])+1000j,cap_cost={0:1000,1:500})["capacitors"].round(1).tolist() , [0,0,1.2,1.2], "optimal sizing failed")
    testEq(test.optimal_sizing(refresh=True,verbose=True,gen_cost=np.array([100,500,1000,1000])+1000j,cap_cost={0:1000,1:500},update_model=True)["additions"] , {'generation': {0: (14.8+0j), 1: (1.6+0j)}, 'capacitors': {2: 1.2}} , "optimal sizing failed")
    testEq(test.optimal_powerflow(refresh=True)["curtailment"].tolist(),[0,0,0,0],"optimal powerflow failed")

    if runtime:
        test.data["globals"]["savefile"] = ""
        test.save("4bus_test.json",indent=4)
        rc,out,err = test.run("4bus_test.json",exception=False)
        testEq(out,[''],'run test failed')

    for case in os.listdir("."):
        if case.startswith("case") and case.endswith(".json"):
            test = Model(case)
            test.optimal_sizing(gen_cost=100,cap_cost=10,update_model=True)
            test.optimal_powerflow(refresh=True)

    test = Model("wecc240.json")
    test.optimal_sizing(gen_cost=100,cap_cost=10,update_model=True,verbose=True)
    print(test.optimal_powerflow(refresh=True))


    print("\nTEST: completed tests",end="",file=sys.stderr,flush=True)
    print("\nTEST:",tested,"tested,",failed,"failed",file=sys.stderr)
