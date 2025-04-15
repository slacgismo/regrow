import sys
import datetime as dt
import gldcore
import gld_pypower
import json

print("Loading",__name__,file=sys.stderr,end="...")

model = None

def on_init(t0):

    print(f"{__name__}.on_init(t0='{dt.datetime.fromtimestamp(t0)}')",file=sys.stderr)
    global model
    model = gld_pypower.Model("wecc240.json")

    print(model.optimal_sizing(),file=sys.stderr)

    return True

def on_precommit(t0):

    print(f"{__name__}.on_precommit(t0='{dt.datetime.fromtimestamp(t0)}')",file=sys.stderr)

    global model
    print(model.optimal_powerflow(),file=sys.stderr)

    return True


print("ok",file=sys.stderr)
