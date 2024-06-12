"""Example controller file for IEEE-14 pypower/gridlabd model"""

import sys
import numpy as np
import scipy as sp
import cvxpy as cp

A = None # unweighted Laplacian matrix
Pd = None
Pg = None
SOC = None
Pmax = None
Tgen = {} 
Tbus = {}

def on_init():
    """on_init() is called when the simulation first starts up

    Return: True on success, False on failure
    """

    # Access a global variable
    loss = gridlabd.get_global("pypower::total_loss")
    # print("loss:",loss,file=sys.stderr)

    # Get a list of objects
    objects = gridlabd.get("objects")
    # print("objects:",objects,file=sys.stderr)

    # Get the properties of an object
    pp_gen_1 = gridlabd.get_object("pp_gen_1")
    # print("pp_gen_1:",pp_gen_1,file=sys.stderr)

    # Get a single object property
    pp_branch_1_current = gridlabd.get_value("pp_branch_1","current")
    # print("pp_branch_1_current:",pp_branch_1_current,file=sys.stderr)

    # Set a single object property (handles unit conversion if needed)
    # gridlabd.set_value("pp_gen_1","Pmax","350e3 kW")

    # # Direct access to a property
    pp_gen_1_Pmax = gridlabd.property("pp_gen_1","Pmax")
    # print("pp_gen_1_Pmax:",pp_gen_1_Pmax.get_value(),file=sys.stderr)
    pp_gen_1_Pmax.set_value(340.0)
    # print("pp_gen_1_Pmax:",pp_gen_1_Pmax.get_value(),file=sys.stderr)

    for obj in objects:
        data = gridlabd.get_object(obj)
        if data["class"] in "powerplant":
            _T = dict(
                S=gridlabd.property(obj,"S"),
                operating_capacity=gridlabd.property(obj,"operating_capacity"),
                storage_capacity=gridlabd.property(obj,"storage_capacity"),
                charging_capacity=gridlabd.property(obj,"charging_capacity"),
                storage_efficiency=gridlabd.property(obj,"storage_efficiency"),
                state_of_charge=gridlabd.property(obj,"state_of_charge"),
                )
            if "parent" in data:
                parent_data = gridlabd.get_object(data["parent"])
                if parent_data["class"] == "gen":
                    Tgen[obj] = _T
                elif parent_data["class"] == "bus":
                    Tbus[obj] = _T
                else:
                    gridlabd.warning(f"object {obj} does not have a bus or gen parent")
            else:
                gridlabd.warning(f"object {obj} does not have a parent")

    # print(Tgen,Tbus,file=sys.stderr)
    return True

def on_precommit(data):
    """TODO"""

    # print(data,file=sys.stderr)

    bus = np.array(data['bus'])
    N = len(bus)
    # print("bus:",bus,file=sys.stderr)

    branch = np.array(data['branch'])
    M = len(branch)
    # print("branch:",branch,file=sys.stderr)

    # impedance values
    Z = np.array([complex(*x) for x in zip(branch.T[2],branch.T[3])])
    # print(Z,file=sys.stderr)

    gen = np.array(data['gen'])
    # print("gen:",gen,file=sys.stderr)

    # get A - Laplacian matrix
    row = [int(x)-1 for x in branch[:,0]]
    col = [int(x)-1 for x in branch[:,1]]
    # print(row,col,file=sys.stderr)
    global A
    A = sp.sparse.coo_array(([1]*len(row) + [-1]*len(row),(row+col,col+row)),(len(bus),len(bus)))
    # print("A:",A.toarray(),file=sys.stderr)

    # SOLVE MPC here (maybe?)


    # get I - weighted line-node incidence matrix
    I = sp.sparse.coo_array((Z,(list(range(M)),row)),shape=(M,N)) - sp.sparse.coo_array((Z,(list(range(M)),col)),shape=(M,N))
    # print("I:",I.toarray(),file=sys.stderr)

    # get L - weighted Laplacian
    L = I.T@I
    # print("L:",L.toarray(),file=sys.stderr)

    # get Pd - demand
    Pd = np.array([bus[:,2]])[0]
    # print("Pd:",Pd,file=sys.stderr)

    # get Pg - generation by gentype in columns
    # get constraints, e.g., Pmax, charger/discharge rate max/min, battery capacities (on_init?)
    Pg = np.zeros(N)
    Pmax = np.zeros(N)
    Pmin = np.zeros(N)
    for n,g,m0,m1 in gen[:,[0,1,8,9]]:
        Pg[int(n)] = g
        # Pmin[int(n)] = m0
        Pmax[int(n)] = m1
    # print("Pg:",Pg,file=sys.stderr)
    # print("Pmax:",Pmax,file=sys.stderr)

    # reference nodes
    ref = [int(x[0]) for x in bus if x[1] == 3]

    # TODO: get energy prices (if any)
    P = np.zeros(N)

    # solve OPF for the upcoming time interval
    x = cp.Variable(N)
    g = cp.Variable(N)
    objective = cp.Minimize(P@g)
    constraints = [
        L.real @ x - g + Pd == 0, # KVL/KCL laws
        x[ref] == 0, # voltage angle of reference bus(es)
        g >= Pmin, # minimum generation capability
        g <= Pmax, # maximum generation capability
        # TODO: add line flow limits (if any)
    ]
    problem = cp.Problem(objective,constraints)
    problem.solve()
    # print(gridlabd.get_global("clock"),"-- OPF is",problem.status,file=sys.stderr)
    if x.value is None:
        gridlabd.warning(f"controllers.on_precommit(t='{gridlabd.get_global('clock')}'): OPF problem is {problem.status}")
    else:

        # post optimal generation dispatch to main solver
        # print(data["bus"])
        _g = g.value.round(3).tolist()
        # print("g.value =",_g,file=sys.stderr)
        for n,gen in gendict.items():
            gen["real"].set_value(_g[n])
            # gen["reactive"].set_value(h.value[n])

    # get SOC - battery state of charge
    
    # store these in global arrays for on_sync to run MPC

    # TODO: add energy storage calcs to powerplant model

    # TODO: figure out forecasting for wind, solar, other generation, and load by peeking at future
    return (int(data['t']/3600)+1)*3600

def on_commit(data):
    """TODO"""
    # get MPC result from global arras
    # put C - charge/discharge rates
    return (int(data['t']/3600)+1)*3600

def on_sync(data):
    """on_sync(data) is called when the clock updates
    
    data (dict) contains all the network model data and simulation context data

    Return: Unix timestamp of next controller event, if any otherwise -1 for
            none, 0 on failure, and data['t'] to iterate again

    See also: `gridlabd --modhelp pypower` for properties in pypower objects
              (i.e., bus, branch, gen, gencost)
    """
    # print(f"controllers sync called, data={data}",file=sys.stderr)
    
    return (int(data['t']/3600)+1)*3600 # advance to top of next hour 

def on_term():
    """on_term() is called when the simulation ends"""
    # print("on_term()",file=sys.stderr)

def load_control(obj,**kwargs):
    """load_control(obj,**kwargs) is called before on_sync()

    obj (str) contains the name of the load
    kwargs (dict) contains the properties of the load

    Return: dict of properties to update, including `t` for next update time

    See also: `gridlabd --modhelp pypower:load`
    """
    # print(obj,": load control update",kwargs,file=sys.stderr)
    
    return dict(t=kwargs['t']+3600) # on_sync() return value to meaning of `t`

def powerplant_control(obj,**kwargs):
    """powerplant_control(obj,**kwargs) is called before on_sync()

    obj (str) contains the properties of the powerplant
    kwargs (dict) contains the properties of the powerplant

    Return: dict of properties to update, including `t` for next update time

    See also: `gridlabd --modhelp pypower:powerplant`
    """
    # print(obj,": powerplant control update",kwargs,file=sys.stderr)
    return dict(t=kwargs['t']+3600)
