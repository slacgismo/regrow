"""Example controller file for IEEE-14 pypower/gridlabd model"""

import sys

def on_init():
    """on_init() is called when the simulation first starts up

    Return: True on success, False on failure
    """
    return True

def on_sync(data):
    """on_sync(data) is called when the clock updates
    
    data (dict) contains all the network model data and simulation context data

    Return: Unix timestamp of next controller event, if any otherwise -1 for
            none, 0 on failure, and data['t'] to iterate again

    See also: `gridlabd --modhelp pypower` for properties in pypower objects
              (i.e., bus, branch, gen, gencost)
    """
    print(f"controllers sync called, data={data}",file=sys.stderr)
    
    return (int(data['t']/3600)+1)*3600 # advance to top of next hour 

def load_control(obj,**kwargs):
    """load_control(obj,**kwargs) is called before on_sync()

    obj (str) contains the name of the load
    kwargs (dict) contains the properties of the load

    Return: dict of properties to update, including `t` for next update time

    See also: `gridlabd --modhelp pypower:load`
    """
    print(obj,": load control update",kwargs,file=sys.stderr)
    
    return dict(t=kwargs['t']+3600, S=(15+2j)) # on_sync() return value to meaning of `t`

def powerplant_control(obj,**kwargs):
    """powerplant_control(obj,**kwargs) is called before on_sync()

    obj (str) contains the properties of the powerplant
    kwargs (dict) contains the properties of the powerplant

    Return: dict of properties to update, including `t` for next update time

    See also: `gridlabd --modhelp pypower:powerplant`
    """
    print(obj,": powerplant control update",kwargs,file=sys.stderr)

    return dict(t=kwargs['t']+3600, S="15+2j kW")
