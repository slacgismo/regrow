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
import os
import sys
opath = sys.path
sys.path.append(os.path.split(__file__)[0])
import model
import pypower
import optimize
sys.path = opath
del opath
