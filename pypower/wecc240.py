"""WECC240 case converter from PSS/E

This converter is designed to be used the same way a case file is used
in PyPOWER, i.e.,

    from wecc240 import wecc240
    case = wecc240()

    from pypower.runpf import runpf
    runpf(case)
"""

from psse import PSSE
from psse2pp import PSSE2PP
from scheduling import Schedule
from datetime import datetime as dt

def wecc240(
    options:list[str]=[],
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

    raw = PSSE("wecc240")
    model = PSSE2PP(raw).model

    if "SCHEDULING" in options:

        schedule = Schedule("wecc240_scheduling_")
        schedule.update_case(model.case)

    if "HIFLD" in options:

        TODO # (future work)

    if "LOADS" in options:

        TODO # (future work)

    if "RENEWABLES" in options:

        TODO # (future work)

    for option in options:
        if not option in ["SCHEDULING","HIFLD","LOADS","RENEWABLES"]:
            raise ValueError(f"{option=} in options is not valid")

    return model.case

