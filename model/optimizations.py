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
    try:
        model = gld_pypower.Model("wecc240.json")
        print(model.optimal_sizing(),file=sys.stderr)
    except:
        gldcore.error("optimal sizing failed")
        model = None


    return True

def on_precommit(t0):

    print(f"{__name__}.on_precommit(t0='{dt.datetime.fromtimestamp(t0)}')",file=sys.stderr)

    global model
    try:
        print(model.optimal_powerflow(update_model=True),file=sys.stderr)
    except:
        gldcore.error("optimal powerflow failed")
        model = None

    return True


print("ok",file=sys.stderr)
