"""Test the wecc240() model conversion from PSS/E to PyPower"""

import os
import sys
from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from psse import PSSE
from psse2pp import PSSE2PP
from wecc240 import wecc240
from ppmodel import PPModel
from pypower.runpf import runpf
from pypower.rundcopf import rundcopf
from pypower.runopf import runopf as runacopf
from pypower.ppoption import ppoption
from pypower import idx_bus as bus

save_plots = False # TODO: enable this

os.makedirs("tests",exist_ok=True)

errors = 0
options = ppoption(VERBOSE=0,OUT_ALL=0)

def time_call(call,*args,**kwargs):
    """Times a call with arguments

    Arguments:

    call: function or method to call

    *args: position arguments to call

    **kwargs: keyword arguments to call

    Returns:

    The return value depends on whether the call returns a singleton
    value or a tuple.

    *tuple, float: return values and execution time in seconds

    value, float: return value and execution time in seconds
    """
    tic = time()
    result = call(*args,**kwargs)
    toc = time()
    if isinstance(result,(tuple,list)):
        return *result,toc-tic
    return result,toc-tic

def plot(basecase:dict,
    testcase:dict,
    prefix:str="",
    ):
    """Generate test result plots

    Arguments:

    basecase: the reference case to use for plotting errors

    testcase: the test case to use for plotting errors

    prefix: plot file prefix to use for saving plots
    """

    # pylint: disable=too-many-instance-attributes

    bus_i = basecase["bus"][:,bus.BUS_I].astype(int).astype(str)
    vm_err = np.array(1-testcase["bus"][:,bus.VM])/np.array(basecase["bus"][:,bus.VM])
    va_err = np.array(testcase["bus"][:,bus.VA])-np.array(basecase["bus"][:,bus.VA])

    #
    # Plot voltage errors by bus
    #
    plt.figure(figsize=(44,28))

    plt.subplot(2,1,1)
    plt.plot(bus_i,basecase["bus"][:,bus.VM],label="Base case")
    plt.plot(bus_i,testcase["bus"][:,bus.VM],label="Test case")
    plt.ylabel("Voltage Magnitude (pu.kV)")
    plt.grid()
    plt.xticks(rotation=90)
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(bus_i,basecase["bus"][:,bus.VA],label="Base case")
    plt.plot(bus_i,testcase["bus"][:,bus.VA]*180/np.pi,label="Test case")
    plt.ylabel("Voltage Angle (deg)")
    plt.xlabel("Bus ID")
    plt.grid()
    plt.xticks(rotation=90)
    plt.legend()

    plt.suptitle(f"{os.path.basename(prefix).replace('_',' ').title()} Voltage")
    plt.savefig(f"{prefix}voltage.png")

    plt.close()

    #
    # Plot voltage errors in order
    #
    data = pd.DataFrame({
        "vm_err":np.abs(vm_err),
        "va_err":np.abs(va_err),
        "bus_i":bus_i}).set_index("bus_i")

    plt.figure(figsize=(20,10))

    ax = plt.subplot(1,2,1)
    (data[["vm_err"]]*100)\
        .sort_values("vm_err",ascending=False)\
        .plot(grid=True,
            ylabel="Voltage Magnitude Error (%)",
            ax=ax,
            legend=False,
            xlabel="Bus rank")

    ax = plt.subplot(1,2,2)
    data[["va_err"]]\
        .sort_values("va_err",ascending=False)\
        .plot(grid=True,
            ylabel="Voltage Angle Error (deg)",
            ax=ax,
            legend=False,
            xlabel="Bus rank")

    plt.suptitle(f"{os.path.basename(prefix).replace('_',' ').title()} Voltage Errors")
    plt.savefig(f"{prefix}voltage_errors.png")

    plt.close()

test_results = pd.DataFrame(data={
    "Model":["Original","Scheduling"],
    "Pre-OPF Powerflow":[float('nan')]*2,
    "DC OPF Solution":[float('nan')]*2,
    "Post-OPF Powerflow":[float('nan')]*2,
    }).set_index("Model")

#
# Test PSSE raw loader
#
raw = PSSE(prefix="wecc240/",raw="wecc240_psse.raw")
assert raw.config["version"] == 34, "incorrect raw file version"
assert raw.config["mvabase"] == 100.0, "incorrect raw mvabase value"
length_checks = {
    "area": 4,
    "bus": 243,
    "branch": 329,
    "gen": 146,
    "load": 135,
    "shunt": 7,
    "xform": 122,
    "zone": 14,
    "dcline": 2,
    "gis": 243,
    "scheduling": 3,
}
for item,length in length_checks.items():
    assert len(getattr(raw,item)) == length, f"incorrect length for raw.{item}, expected {length}, found {len(getattr(raw,item))}"

#
# Test PSSE to PyPOWER converter
#
pp = PSSE2PP(raw)
assert hasattr(pp,"model"), "model not found in PSSE2PP converter"
assert hasattr(pp.model,"case"), "case not found in PSSE2PP converter model"
length_checks = {
    "bus": 243,
    "branch": 451,
    "gen": 146,
    "gencost": 146,
    "dcline": 2,
    "dclinecost": 2,
    "gis": 243,
    "scheduling": 1, # TODO: should be 3 when scheduling of line and storage is done
}
for item,length in length_checks.items():
    assert len(pp.model.case[item]) == length, f"incorrect length for model {item}, expected {length}, found {len(pp.model.case[item])}"

#
# Test PPMOdel accessors
#
length_checks = {
    "Bus count": 243,
    "Branch count": 451,
    "Generator count": 146,
    "DC line count": 2,
    "Node count": 126,
    "LV busses": 56,
    "MV busses": 92,
    "HV busses": 95,
    "Generation substations": 55,
    "Load substations": 115,
}
info = pp.model.get_info()
for item,length in length_checks.items():
    assert info[item] == length, f"incorrect length for model.get_info('{item}'), expected {length}, found {info[item]}"
length_checks = {
    "bus": 243,
    "branch": 451,
    "gen": 146,
    "gencost": 146,
    "dcline": 2,
    "dclinecost": 2,
    "gis": 243,
}
for item,length in length_checks.items():
    assert len(pp.model.get_data(item)) == length, f"incorrect length for PPModel.get_bus('{item}'), expected {length}, found {len(pp.model.case[item])}"

#
# Verify the original WECC 240 model
#

# load the model
PSSE2PP.LOADSCALE = 1.0 # global scaling of loads
original = wecc240()

# save the case data
with open("tests/wecc240_original.py","w",encoding="utf-8") as fh:
    PPModel("wecc240",case=original).save_case(fh)

    # solve the original powerflow from PSSE
    print(f"Running runpf of {fh.name}...")
    original_solution,status,xtime = time_call(runpf,original,options)
    if status == 0:
        print(f"ERROR [wecc240]: original case powerflow failed (see {fh.name})")
        errors += 1
    else:
        # print(f"Original WECC240 powerflow solved in {xtime:.3f} seconds.",flush=True)
        test_results.loc["Original","Pre-OPF Powerflow"] = xtime

    if save_plots:
        print("Saving comparison plots to tests folder",end="...",flush=True)
        plot(basecase=original,testcase=original_solution,prefix="tests/original_")
        plot(basecase=original,testcase=original_solution,prefix="tests/original_")
        print("done")

    # solve the original model DCOPF
    print(f"Running rundcopf of {fh.name}...")
    dcopf,xtime = time_call(rundcopf,original,options)
    if not dcopf["success"]:
        print(f"ERROR [wecc240]: original case dcopf failed (see {fh.name})")
        errors += 1
    else:
        # print(f"Original WECC240 DC OPF solved in {xtime:.3f} seconds.",flush=True)
        test_results.loc["Original","DC OPF Solution"] = xtime

# solve the DCOPF powerflow
with open("tests/wecc240_original_dcopf.py","w",encoding="utf-8") as fh:
    PPModel("wecc240",case=dcopf).save_case(fh)
    print(f"Running runpf of {fh.name}...")
    dcopf_solution,status,xtime = time_call(runpf,dcopf,options)
    if status == 0:
        print(f"ERROR [wecc240]: original case dcopf powerflow failed (see {fh.name})")
        errors += 1
    else:
        # print(f"WECC240 DC OPF powerflow solved in {xtime:.3f} seconds.",flush=True)
        test_results.loc["Original","Post-OPF Powerflow"] = xtime

    if save_plots:
        print("Saving comparison plots to tests folder",end="...",flush=True)
        plot(basecase=original,testcase=dcopf_solution,prefix="tests/original_dcopf_")
        plot(basecase=original,testcase=dcopf_solution,prefix="tests/original_dcopf_")
        print("done")


#
# Verify the scheduling WECC 240 model
#
scheduling = wecc240(options=["SCHEDULING"])
with open("tests/wecc240_scheduling.py","w",encoding="utf-8") as fh:
    PPModel("wecc240",case=scheduling).save_case(fh)

    # solve the scheduling powerflow from PSSE
    print(f"Running runpf of {fh.name}...")
    scheduling_solution,status,xtime = time_call(runpf,scheduling,options)
    if status == 0:
        print(f"ERROR [wecc240]: scheduling case powerflow failed (see {fh.name})")
        errors += 1
    else:
        # print(f"Scheduling WECC240 powerflow solved in {xtime:.3f} seconds",flush=True)
        test_results.loc["Scheduling","Pre-OPF Powerflow"] = xtime

    # solve the schedule model DCOPF
    print(f"Running rundcopf of {fh.name}...")
    scheduling_dcopf,xtime = time_call(rundcopf,scheduling,options)
    if not scheduling_dcopf["success"]:
        print(f"ERROR [wecc240]: scheduling case dcopf failed (see {fh.name})")
        errors += 1
    else:
        # print(f"Scheduling WECC240 DC OPF solved in {xtime:.3f} seconds.",flush=True)
        test_results.loc["Scheduling","DC OPF Solution"] = xtime

    print(f"Running runacopf of {fh.name}...")
    scheduling_acopf,xtime = time_call(runacopf,scheduling,ppoption(OUT_ALL=1))
    if not scheduling_acopf["success"]:
        print(f"ERROR [wecc240]: scheduling case acopf failed (see {fh.name})")
        errors += 1
    else:
        print(f"Schedule WECC240 AC OPF solved in {xtime:.3f} seconds.",flush=True)

# solve the DCOPF powerflow
with open("tests/wecc240_scheduling_dcopf.py","w",encoding="utf-8") as fh:
    PPModel("wecc240",case=scheduling_dcopf).save_case(fh)
    print(f"Running runpf of {fh.name}...")
    scheduling_dcopf_solution,status,xtime = time_call(runpf,scheduling_dcopf,options)
    if status == 0:
        print(f"ERROR [wecc240]: scheduling case dcopf powerflow failed (see {fh.name})")
        errors += 1
    else:
        # print(f"WECC240 scheduling DC OPF powerflow solved in {xtime:.3f} seconds.",flush=True)
        test_results.loc["Scheduling","Post-OPF Powerflow"] = xtime

    if save_plots:
        print("Saving comparison plots to tests folder",end="...",flush=True)
        plot(basecase=original,testcase=scheduling_dcopf_solution,prefix="tests/scheduling_dcopf_")
        plot(basecase=original,testcase=scheduling_dcopf_solution,prefix="tests/scheduling_dcopf_")
        print("done")

#
# Test save kml files
#
if save_plots:
    print("Saving KML files to tests folder",end="...")
    PPModel("wecc240",case=original).save_kml("tests/wecc240_original.kml")
    PPModel("wecc240",case=scheduling).save_kml("tests/wecc240_scheduling.kml")
    print("done")

if errors > 0:
    print(f"WECC240 failed {errors} test.")
else:
    print("Test solution time (ms)")
    print("-----------------------")
    print((test_results.round(3)*1000).astype(int))

sys.exit(errors)
