"""Example controller file for IEEE-14 pypower/gridlabd model"""

import sys
import numpy as np
import gridlabd

A = None
Pd = None
Pg = None
SOC = None
Pmax = None

def on_init():
    """on_init() is called when the simulation first starts up

    Return: True on success, False on failure
    """
    return True

def on_precommit(data):
    """TODO"""

    # print(data,file=sys.stderr)

    bus = np.array(data['bus'])
    # print(bus,file=sys.stderr)

    branch = np.array(data['branch'])
    # print(branch,file=sys.stderr)

    gen = np.array(data['gen'])
    # print(gen,file=sys.stderr)

    # get A - Laplacian matrix
    global A
    # TODO: rewrite using sparse matrix
    A = np.zeros((len(bus),len(bus)))
    for fbus,tbus in [(int(x)-1,int(y)-1) for x,y in branch[:,0:2]]:
        A[fbus,tbus] = 1
        A[tbus,fbus] = -1
    # print(A,file=sys.stderr)

    # get Pd - demand
    Pd = np.array([bus[:,2]]).transpose()
    # print(Pd,file=sys.stderr)

    # get Pg - generation by gentype in columns
    # get constraints, e.g., Pmax, charger/discharge rate max/min, battery capacities (on_init?)
    Pg = np.zeros((len(bus),1))
    Pmax = np.zeros((len(bus),1))
    for n,g,m in gen[:,[0,1,8]]:
        Pg[int(n)] = g
        Pmax[int(n)] = m
    # print(Pg,file=sys.stderr)
    # print(Pmax,file=sys.stderr)


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
