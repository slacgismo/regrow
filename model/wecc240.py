import recorders
import optimal_sizing

def on_init(t0):
    return recorders.on_init(t0) and optimal_sizing.on_init(t0)

def on_commit(t0):
    return recorders.on_commit(t0)
