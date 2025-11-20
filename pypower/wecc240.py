"""WECC240 case converter from PSS/E

This converter is designed to be used the same way a case file is used
in PyPOWER, i.e.,

    from wecc240 import wecc240
    case = wecc240()

    from pypower.runpf import runpf
    runpf(case)

Note that although the return value of the `wecc240()` method is a PyPower
case, this script is not a PyPower case file.  Use the `save_case()` method
of the PPModel class to save the case file, e.g.,

    from ppmodel import PPModel
    PPModel("wecc240").set_case(case).save_case("my_cases/wecc240.py")

WARNING: Be careful not to overwrite this file when saving the case data
returned by this module.
"""

from datetime import datetime as dt
import pandas as pd
from psse import PSSE
from psse2pp import PSSE2PP
from scheduling import Schedule
from hifld import HIFLD, WECC
from pypower import idx_bus

def wecc240(
    options:list[str]=None,
    datetime:dt=None,
    ) -> dict:
    """Load and convert the WECC 240 PSSE RAW model to a PyPOWER case

    Argument:

    options: model extension options
        "SCHEDULING": include WECC240 scheduling data
        "HIFLD": include HIFLD generation fleet data (future work)
        "LOADS": include NREL demand model (future work)
        "RENEWABLES": include NREL renewable generation fleet (future work)

    datetime: date and time at which to update LOADS and RENEWABLES, if specified

    Returns:

    dict: pypower case (see PyPower documentation)
    """

    # load the model from PSSE
    psse = PSSE("wecc240/","wecc240_psse.raw")
    model = PSSE2PP(psse).model

    # default is no options
    if options is None:
        options = []

    # check for improperly specified options
    for option in options:
        if not option in ["SCHEDULING","HIFLD","LOADS","RENEWABLES"]:
            raise ValueError(f"{option=} in options is not valid")

    # process options
    if "SCHEDULING" in options:

        schedule = Schedule("wecc240/scheduling/")
        schedule.update_case(model.case)

    if "HIFLD" in options:

        # select only busses with 20 kV powerplant substations
        busses = pd.merge(psse.gis,psse.bus,left_on="BUS_I",right_on="ID")
        busses.drop("ID",axis=1,inplace=True)
        busses.drop(busses[busses.BASEKV>20.0].index,inplace=True)

        # load HIFLD operating non-renewable WECC powerplants summed by bus and type
        powerplants = HIFLD(
            drop_test=lambda x: x[(~x["STATE"].isin(WECC))|(x["STATUS"]!="OP")].index,
            drop_types=["PV","WT","UNKNOWN"],
            drop_fuels=["SUN","WIND"],
            busdata=busses[busses["BASEKV"]==20],
            groupby=["BUSCODE","TYPE"],
            )

        raise NotImplementedError("TODO")

    if "LOADS" in options:

        raise NotImplementedError("TODO")

    if "RENEWABLES" in options:

        raise NotImplementedError("TODO")

    return model.case

if __name__ == "__main__":

    case = wecc240(["HIFLD"])
