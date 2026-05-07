import cvxpy as cp
import numpy as np
from tqdm import tqdm

from .constraints import (
    battery_dynamics_contraints,
    conservation_of_power_constraints,
    validate_battery_dynamics,
    validate_solution_dynamics,
)
from .metrics import core_objective


def _make_mpc_subproblem(H, Q, B, G, alpha, beta, lamb, mu, gamma, efficiency, soc_loss, q_target, delta):
    """make an MPC subproblem with horizon H"""

    # battery param
    param_q0 = cp.Parameter(nonneg=True, name="q0")  # battery  starting SOC

    # data parameters
    l = cp.Parameter(H, name="l")
    R = cp.Parameter(H, name="R")

    # variables
    g = cp.Variable(H, nonneg=True, name="g")  # dispatchable gen
    r = cp.Variable(H, nonneg=True, name="r")  # non-dispatachable gen
    c = cp.Variable(H, nonneg=True, name="c")  # curtailed non-dispatachable gen
    b = cp.Variable(H, name="b")  # battery power
    b_out = cp.Variable(H, nonneg=True, name="b_out")  # battery discharge
    b_in = cp.Variable(H, nonneg=True, name="b_in")  # battery charge
    y = cp.Variable(H, nonneg=True, name="y")  # battery dynamics helper
    s = cp.Variable(H, nonneg=True, name="s")  # load shedding
    q = cp.Variable(H + 1, nonneg=True, name="q")  # battery SOC

    # form problem
    battery_dynamics_constraints = battery_dynamics_contraints(
        q=q,
        q0=param_q0,
        Q=Q,
        b=b,
        b_out=b_out,
        b_in=b_in,
        y=y,
        B=B,
        charge_efficiency=efficiency,
        discharge_efficiency_inv=1 / efficiency,
        soc_loss=soc_loss,
        delta=delta,
    )
    power_constraints = conservation_of_power_constraints(g=g, G=G, r=r, R=R, b=b, l=l, s=s, c=c)
    constraints = battery_dynamics_constraints + power_constraints
    core_obj = core_objective(s=s, g=g, b=b, lamb=lamb, alpha=alpha, beta=beta, mu=mu)
    objective = core_obj + gamma * cp.square(q[-1] - q_target * Q)
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem


def _set_mpc_subproblem_data_params(problem, t, h, l, R, q0):
    pd = problem.param_dict
    pd["l"].value = l[t : t + h]
    pd["R"].value = R[t : t + h]
    pd["q0"].value = q0


def run_mpc_perfect(
    l,
    R,
    G,
    Q,
    B,
    alpha,
    beta,
    lamb,
    gamma,
    mu,
    q_init,
    q_target,
    H,
    efficiency=0.98,
    soc_loss=0,
    delta=1,
    solver="CLARABEL",
    disable_progress_bar=False,
):
    """
    MPC with perfect information of the next H time steps

    args:
        l: load time series
        R: non-dispatachable generation time series
        G: max dispatchable generation
        Q: battery capacity
        B: max battery power
        alpha: dispatchable gen linear cost
        beta: dispatchable gen quadratic cost
        lamb: load shedding penalty
        gamma: penalty strength for deviating from end of horizon soc target
        mu: battery degradation penalty
        q_target: target_soc at end of horizon
        efficiency: charge/discharge efficiency
        soc_loss: battery SOC loss rate per hour
        H: number of timesteps in horizon
        delta: timesteps in an hour
        solver: solver to call

    returns:
        dict of implemented variable trajectories
    """
    T = len(l)
    assert 1 <= H <= T, f"horizon H={H} must be between 1 and T={T} (inclusive)"
    g_traj = np.zeros(T)
    r_traj = np.zeros(T)
    c_traj = np.zeros(T)
    b_traj = np.zeros(T)
    b_out_traj = np.zeros(T)
    b_in_traj = np.zeros(T)
    s_traj = np.zeros(T)
    y_traj = np.zeros(T)
    q_traj = np.zeros(T + 1)
    q_traj[0] = q_init

    mpc_subproblem = _make_mpc_subproblem(
        H,
        Q=Q,
        B=B,
        G=G,
        alpha=alpha,
        beta=beta,
        lamb=lamb,
        mu=mu,
        gamma=gamma,
        efficiency=efficiency,
        soc_loss=soc_loss,
        q_target=q_target,
        delta=delta,
    )
    for t in tqdm(range(T), disable=disable_progress_bar):
        h = min(H, T - t)

        # make a new subproblem to accomodate shorter horizon
        if h < H:
            mpc_subproblem = _make_mpc_subproblem(
                h,
                Q=Q,
                B=B,
                G=G,
                alpha=alpha,
                beta=beta,
                lamb=lamb,
                mu=mu,
                gamma=gamma,
                efficiency=efficiency,
                soc_loss=soc_loss,
                q_target=q_target,
                delta=delta,
            )

        # update data and solve subproblem
        q0 = q_traj[t]
        _set_mpc_subproblem_data_params(problem=mpc_subproblem, t=t, h=h, l=l, R=R, q0=q0)
        mpc_subproblem.solve(solver=solver, warm_start=True)
        solution_valid = validate_solution_dynamics(
            mpc_subproblem, Q=Q, B=B, efficiency=efficiency, soc_loss=soc_loss, delta=delta
        )
        if not solution_valid:
            raise ValueError(f"MPC subproblem solution at timestep t={t} has invalid dynamics")

        # implement the actions prescribed for the immediate time step
        subproblem_solution = mpc_subproblem.var_dict
        g_traj[t] = subproblem_solution["g"].value[0]
        r_traj[t] = subproblem_solution["r"].value[0]
        c_traj[t] = subproblem_solution["c"].value[0]
        b_traj[t] = subproblem_solution["b"].value[0]
        b_out_traj[t] = subproblem_solution["b_out"].value[0]
        b_in_traj[t] = subproblem_solution["b_in"].value[0]
        s_traj[t] = subproblem_solution["s"].value[0]
        y_traj[t] = subproblem_solution["y"].value[0]
        q_traj[t + 1] = subproblem_solution["q"].value[1]

    solution = {
        "g": g_traj,
        "r": r_traj,
        "c": c_traj,
        "b": b_traj,
        "b_out": b_out_traj,
        "b_in": b_in_traj,
        "s": s_traj,
        "y": y_traj,
        "q": q_traj,
    }
    mpc_trajectory_valid = validate_battery_dynamics(
        q=q_traj,
        b=b_traj,
        b_out=b_out_traj,
        b_in=b_in_traj,
        y=y_traj,
        Q=Q,
        B=B,
        q0=q_init,
        charge_efficiency=efficiency,
        discharge_efficiency_inv=1 / efficiency,
        soc_loss=soc_loss,
        delta=delta,
    )
    if not mpc_trajectory_valid:
        raise ValueError("Implemented MPC trajectory has invalid dynamics")
    return solution
