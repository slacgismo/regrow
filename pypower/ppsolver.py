"""PyPOWER solver

This model implements the PyPOWER static and timeseries
solvers.

Example:
    from wecc240 import wecc240
    model = PPModel(case=wecc240)

    solver = PPSolver(model)
    solver.run_timeseries(
        start="2020-08-01 00:00:00+07:00",
        end="2020-09-01 00:00:00+07:00",
        freq="1h",
        )
"""

import datetime as dt
from time import time
from typing import Callable
import warnings

import pandas as pd

from pypower.runpf import runpf
from pypower.rundcopf import rundcopf
from pypower.runopf import runopf as runacopf
from pypower.ppoption import ppoption

class PPSolver:
    """PyPOWER solver implementation"""
    def __init__(self,model):
        """PyPOWER model solver"""
        self.model = model

    def solve_pf(self,
        update:str='success',
        with_result:bool=False,
        ) -> bool:
        """Solve the powerflow problem

        Arguments:

        update: when to update of model case data ('always','success','failure')

        with_result: include result in return value

        Returns:

        bool: True on success, False on failure

        dict: result (if with_result is True)
        """
        assert update in ["always","success","failure"], f"{update=} is invalid"
        result,status = runpf(self.model.case,ppoption(**self.model.options))
        success = status == 1
        if ( success and update in ["always","success"] ) \
                or ( not success and update in ["always","failure"] ):
            for name,values in result.items():
                if name in self.model.case:
                    self.model.case[name] = values
        if with_result:
            return status==1,result
        return status==1

    def solve_opf(self,
        use_acopf:bool=False,
        update:str='success',
        with_result:bool=False,
        ):
        """Solve the optimal powerflow problem

        Arguments:

        use_acopf: enable AC OPF solution

        update: when update of model case data ('always','success','failure')
        
        with_result: include result in return value

        Returns:

        bool: True on success, False on failure

        dict: result (if with_result is True)
        """
        assert use_acopf in [True,False], f"{use_acopf=} is invalid"
        assert update in ["always","success","failure"], f"{update=} is invalid"
        opf = (runacopf if use_acopf else rundcopf)
        result = opf(self.model.case,ppoption(**self.model.options))
        success = result["success"] is True
        if ( success and update in ["always","success"] ) \
                or ( not success and update in ["always","failure"] ):
            for name,values in result.items():
                if name in self.model.case:
                    self.model.case[name] = values
            self.model.case["cost"] = result["f"]
        if with_result:
            return success,result
        return success

    def update_inputs(self,t:dt.datetime):
        """Synchronize inputs with the current date/time

        Arguments:

        t: the current date/time
        """
        # update inputs
        errors = 0
        for name,spec in self.model.inputs.items():
            data = spec["data"]
            name,column = name
            column_number = self.model.get_header(name).index(column)
            mapping = spec["mapping"]["index"]
            scales = spec["mapping"]["scale"]
            try:
                target = self.model.case[name]
                target[mapping,column_number] = data.loc[t] * scales
            except KeyError as exception:
                warnings.warn(f"input({name=},{column=}) {exception=}")
                errors += 1
        return errors

    def update_outputs(self,
        t:dt.datetime,
        ts_format:str="%Y-%m-%d %H:%M:%S %Z"
        ):
        """Synchronize outputs to the current date/time

        Arguments:

        t: the current date/time

        ts_format: timestamp format
        """
        ts = t.strftime(ts_format)

        # output
        for file,spec in self.model.outputs.items():
            mapping = spec["mapping"]
            scale = mapping["scale"]
            offset = mapping["offset"]
            data = [f"{{0:{spec['format']}}}".format(x)
                for x in self.model.get_data(
                    spec["name"]).loc[mapping["rows"],spec["column"]]*scale + offset]
            print(ts,*data,sep=",",file=spec["fh"],flush=True)

        # recorders
        errors = 0
        for file,recorder in self.model.recorders.items():
            values = [ts]
            for _,spec in recorder["targets"].items():
                value = self.model.case
                for level in spec["source"]:
                    if not level in value:
                        warnings.warn(f"recorder '{file}' -- '{level}' not found")
                        errors += 1
                        break
                    value = value[level]
                if not isinstance(value,(int,float,bool,str,type(None))):
                    value = float('nan')
                elif isinstance(value,(int,float)):
                    value = spec["transform"](value)
                values.append(f"{{0:{spec['format']}}}".format(value))
            print(*values,sep=",",file=recorder["fh"],flush=True)
        return errors

    def run_timeseries(self,*args,
        # pylint: disable=too-many-arguments,too-many-locals
        progress:Callable=None,
        call_on_fail:Callable=None,
        stop_on_fail:bool=True,
        stop_test:Callable=None,
        use_acopf:bool=False,
        **kwargs):
        """Run a timeseries simulation

        Arguments:

        *args, **kwargs: See pandas.date_range()

        progress: set a progress callback function

        call_on_fail: set a call-on-fail function

        stop_on_fail: enable stop-on-fail condition

        stop_test: set a stop test call back function

        use_acopf: enable use of AC OPF instead of DC OPF

        Returns:

        None: No errors to report

        str: Error message (when stop_on_fail is True)

        list[str]: Error messages (when stop_on_fail is False)
        """

        assert progress is None or callable(progress), \
            "progress must be callable or None"
        assert call_on_fail is None or callable(call_on_fail), \
            "call_on_fail must be callable or None"
        assert stop_test is None or callable(stop_test), \
            "stop_test must be callable or None"
        assert isinstance(stop_on_fail,bool), \
            "stop_on_fail must be boolean"
        assert isinstance(use_acopf,bool), \
            "use_acopf must be boolean"

        self.model.errors = [] # collect errors, if any
        tic0 = time()

        # process time specified range
        trange = pd.date_range(*args,**kwargs)
        niters = 0
        topf = 0.0
        tpf = 0.0

        # start recorders
        for file,recorder in self.model.recorders.items():
            recorder["fh"] = open(file,"w",encoding="utf-8")
            columns = ["timestamp"] + list(recorder["targets"].keys())
            print(*columns,sep=",",file=recorder["fh"],flush=True)

        # start outputs
        for file,output in self.model.outputs.items():
            output["fh"] = open(file,"w",encoding="utf-8")
            columns = ["timestamp"] + output["mapping"]["columns"]
            print(*columns,sep=",",file=output["fh"],flush=True)

        for t in (x.tz_convert("UTC") for x in trange):

            # setup time and progress/stop callback
            ts = t.strftime("%Y-%m-%d %H:%M:%S %Z")
            if callable(progress) and progress(f"""{ts} ({len(self.model.errors)
                    if self.model.errors else 'no'} errors)"""):
                return None

            # update inputs
            if self.update_inputs(t) > 0:
                if call_on_fail:
                    call_on_fail("Input error")
                if stop_on_fail:
                    break

            # solve OPF and check result
            status,result = self.solve_opf(use_acopf,with_result=True)
            if status is not True:
                failed = f"OPF failed at {ts}"
                self.model.errors.append(failed)
                if call_on_fail:
                    call_on_fail(failed)
                if stop_on_fail:
                    break
            topf += result["et"]

            # solver powerflow and check result
            status,result = self.solve_pf(with_result=True)
            if status is not True:
                failed = f"PF failed at {ts}"
                self.model.errors.append(failed)
                if call_on_fail:
                    call_on_fail(failed)
                if stop_on_fail:
                    break
            tpf += result["et"]

            # process outputs
            if self.update_outputs(t) > 0:
                if call_on_fail:
                    call_on_fail("Output error")
                if stop_on_fail:
                    break

            # check stop condition
            niters += 1
            if stop_test and stop_test(t):
                self.model.errors.append(f"Stopped at {t=}")
                break

        ttot = time() - tic0
        self.model.profile = {
            "Expected iterations": len(trange),
            "Completed iterations": niters,
            "Total OPF time (s)": round(topf,4),
            "Fraction OPF time (s/s)": round(topf/ttot,2) if ttot > 0 else 0,
            "Total PF time (s)": round(tpf,4),
            "Fraction PF time (s/s)": round(tpf/ttot,2) if ttot > 0 else 0,
            "Total run time (s)": round(ttot,4),
            "Iteration time (s/iter)": round(ttot/niters,4) if niters > 0 else "N/A",
        }

        return self.model.errors if self.model.errors else None

if __name__ == "__main__":

    from ppmodel import PPModel
    from ppdata import PPData
    from wecc240 import wecc240
    import pytz

    test_model = PPModel(case=wecc240)

    test_data = PPData(test_model)

    test_data.set_input("bus","PD","tests/load.csv",scale=10)
    test_data.set_input("bus","QD","tests/load.csv",scale=1)

    test_data.set_output("bus","VM","results/bus_vm.csv",formatting=".3f")
    test_data.set_output("bus","VA","results/bus_va.csv",formatting=".4f")
    test_data.set_output("bus","PD","results/bus_pd.csv",formatting=".4f")
    test_data.set_output("bus","QD","results/bus_qd.csv",formatting=".4f")

    test_data.set_recorder("results/cost.csv","cost",["cost"],
        scale=test_model.case['baseMVA'],formatting=".2f")
    test_data.set_recorder("results/cost.csv","cost_pumva",["cost"],
        formatting=".2f")

    start = dt.datetime(2020,7,31,17,0,0,0,pytz.UTC)
    end = dt.datetime(2020,8,31,16,0,0,0,pytz.UTC)

    test_solver = PPSolver(test_model)

    print(f"Running timeseries solution from {start} to {end}",end="")
    test_solver.run_timeseries(
        start=start,
        end=end,
        freq="1h",
        progress=lambda x:print(flush=True,end="."),
        stop_on_fail=False,
        )

    print("done")
    print(pd.DataFrame({"Value":test_model.profile.values()},test_model.profile.keys()))
