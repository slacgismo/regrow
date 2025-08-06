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
import io
import subprocess
import numpy as np
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
    on_invalid:callable=_problem_invalid,
    on_fail:callable=_solver_failed,
    **kwargs) -> dict:
    """Compute optimal powerflow

    Arguments:
    * refresh: force recalculation of previous result
    * verbose: output solver data and results
    * curtailment_price: price at which load is curtailed
    * ref: reference bus id or name
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
            cp.abs(y - 1) <= 0.05,  # limit voltage magnitude to 5% deviation
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
            cp.abs(y - 1) <= 0.05,  # limit voltage magnitude to 5% deviation
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

    return self.set_result("optimal_sizing",result)
