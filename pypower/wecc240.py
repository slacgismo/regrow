"""WECC240 case converter from PSS/E

This converter is designed to be used the same way a case file is used
in PyPOWER, i.e.,

    from wecc240 import wecc240
    case = wecc240()

    from pypower.runpf import runpf
    runpf(case)
"""

from datetime import datetime as dt
from psse import PSSE
from psse2pp import PSSE2PP
from scheduling import Schedule

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
    model = PSSE2PP(PSSE("wecc240")).model

    # default is no options
    if options is None:
        options = []

    # check for improperly specified options
    for option in options:
        if not option in ["SCHEDULING","HIFLD","LOADS","RENEWABLES"]:
            raise ValueError(f"{option=} in options is not valid")

    # process options
    if "SCHEDULING" in options:

        schedule = Schedule("wecc240_scheduling_")
        schedule.update_case(model.case)

    if "HIFLD" in options:

        raise NotImplementedError("future work")

    if "LOADS" in options:

        raise NotImplementedError("future work")

    if "RENEWABLES" in options:

        raise NotImplementedError("future work")

    return model.case
