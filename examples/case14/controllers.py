"""Example controller file for IEEE-14 pypower/gridlabd model"""

import sys
import numpy as np
import scipy as sp

A = None
Pd = None
Pg = None
SOC = None
Pmax = None

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

    return True

def on_precommit(data):
    """TODO"""

    # print(data,file=sys.stderr)

    bus = np.array(data['bus'])
    # print("bus:",bus,file=sys.stderr)

    branch = np.array(data['branch'])
    # print("branch:",branch,file=sys.stderr)

    gen = np.array(data['gen'])
    # print("gen:",gen,file=sys.stderr)

    # get A - Laplacian matrix
    global A
    row = [int(x)-1 for x in branch[:,0]]
    col = [int(x)-1 for x in branch[:,1]]
    A = sp.sparse.coo_array(([1]*len(row) + [-1]*len(row),(row+col,col+row)),(len(bus),len(bus)))
    # print("A:",A.toarray(),file=sys.stderr)

    # get Pd - demand
    Pd = np.array([bus[:,2]])[0]
    # print("Pd:",Pd,file=sys.stderr)

    # get Pg - generation by gentype in columns
    # get constraints, e.g., Pmax, charger/discharge rate max/min, battery capacities (on_init?)
    Pg = np.zeros((len(bus)))
    Pmax = np.zeros((len(bus)))
    for n,g,m in gen[:,[0,1,8]]:
        Pg[int(n)] = g
        Pmax[int(n)] = m
    # print("Pg:",Pg,file=sys.stderr)
    # print("Pmax:",Pmax,file=sys.stderr)


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
