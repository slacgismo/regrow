import cvxpy as cp

from .constraints import (
    battery_dynamics_contraints,
    conservation_of_power_constraints,
)
from .metrics import core_objective


def make_one_shot(alpha, beta, lamb, mu, T, delta=1):
    """
    make the perfect knowledge one shot battery optimal control problem

    args:
        T: number of time steps
        delta: number of hours per time step

    return:
        problem: the cvxpy problem object
    """
    # battery and controller params
    param_Q = cp.Parameter(nonneg=True, name="Q")  # battery capacity
    param_q0 = cp.Parameter(nonneg=True, name="q0")
    param_B = cp.Parameter(nonneg=True, name="B")  # max power
    param_charge_efficiency = cp.Parameter(nonneg=True, name="charge_efficiency")
    param_discharge_efficiency_inv = cp.Parameter(nonneg=True, name="discharge_efficiency_inv")
    param_soc_loss = cp.Parameter(nonneg=True, name="soc_loss_per_hour")  # battery SOC loss rate per hour

    # data params
    param_l = cp.Parameter(T, name="l")
    param_R = cp.Parameter(T, name="R")
    param_G = cp.Parameter(1, name="G")

    # variables
    g = cp.Variable(T, nonneg=True, name="g")  # dispatchable gen
    r = cp.Variable(T, nonneg=True, name="r")  # non-dispatachable gen
    c = cp.Variable(T, nonneg=True, name="c")  # curtailed non-dispatachable gen
    b = cp.Variable(T, name="b")  # battery power
    b_out = cp.Variable(T, nonneg=True, name="b_out")  # battery discharge
    b_in = cp.Variable(T, nonneg=True, name="b_in")  # battery charge
    y = cp.Variable(T, nonneg=True, name="y")  # battery dynamics helper variable
    s = cp.Variable(T, nonneg=True, name="s")  # load shedding
    q = cp.Variable(T + 1, nonneg=True, name="q")  # battery SOC

    # form problem
    battery_dynamics_constraints = battery_dynamics_contraints(
        q=q,
        q0=param_q0,
        Q=param_Q,
        b=b,
        b_out=b_out,
        b_in=b_in,
        y=y,
        B=param_B,
        charge_efficiency=param_charge_efficiency,
        discharge_efficiency_inv=param_discharge_efficiency_inv,
        soc_loss=param_soc_loss,
        delta=delta,
    )
    power_constraints = conservation_of_power_constraints(g=g, G=param_G, r=r, R=param_R, b=b, l=param_l, s=s, c=c)
    constraints = battery_dynamics_constraints + power_constraints
    objective = core_objective(
        s=s,
        g=g,
        b=b,
        lamb=lamb,
        alpha=alpha,
        beta=beta,
        mu=mu,
    )
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem


def load_one_shot_problem_data(problem, l, R, G, Q, B, efficiency, soc_loss):
    pd = problem.param_dict
    pd["l"].value = l
    pd["R"].value = R
    pd["G"].value = G
    pd["Q"].value = Q
    pd["B"].value = B
    pd["charge_efficiency"].value = efficiency
    pd["discharge_efficiency_inv"].value = 1 / efficiency  # assume discharge efficiency same as charge efficiency
    pd["soc_loss_per_hour"].value = soc_loss
