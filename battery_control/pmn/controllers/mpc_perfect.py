import cvxpy as cp
import numpy as np
from constraints import (
    battery_dynamics_contraints,
    conservation_of_power_constraints,
    validate_battery_dynamics,
    validate_solution_dynamics,
)


def _make_mpc_subproblem(H, delta):
    """make an MPC subproblem with horizon H"""

    # battery and controller params
    param_Q = cp.Parameter(nonneg=True, name="Q")  # battery size
    param_B = cp.Parameter(nonneg=True, name="B")  # max power
    param_alpha = cp.Parameter(nonneg=True, name="alpha")  # dispatchabale gen linear cost
    param_beta = cp.Parameter(nonneg=True, name="beta")  # dispatachable gen quadratic cost
    param_lambda = cp.Parameter(nonneg=True, name="lambda")  # load shedding penalty paramater
    param_mu = cp.Parameter(nonneg=True, name="mu")  # battery degradation penalty term
    param_gamma = cp.Parameter(nonneg=True, name="gamma")  # MPC terminal target term strength
    param_charge_efficiency = cp.Parameter(nonneg=True, name="charge_efficiency")
    param_discharge_efficiency_inv = cp.Parameter(
        nonneg=True, name="discharge_efficiency_inv"
    )  # use inverse to comply with DPP
    param_soc_loss = cp.Parameter(nonneg=True, name="soc_loss_per_hour")  # battery SOC loss rate per hour
    param_q0 = cp.Parameter(nonneg=True, name="q0")  # battery  starting SOC
    param_q_target = cp.Parameter(nonneg=True, name="q_target")

    # data parameters
    l = cp.Parameter(H, name="l")
    R = cp.Parameter(H, name="R")
    G = cp.Parameter(nonneg=True, name="G")

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
        Q=param_Q,
        b=b,
        b_out=b_out,
        b_in=b_in,
        y=y,
        B=param_B,
        charge_efficiency=param_charge_efficiency,
        dishcarge_efficiency_inv=param_discharge_efficiency_inv,
        soc_loss=param_soc_loss,
        delta=delta,
    )
    power_constraints = conservation_of_power_constraints(g=g, G=G, r=r, R=R, b=b, l=l, s=s, c=c)
    constraints = battery_dynamics_constraints + power_constraints
    objective = 1 / H * cp.sum(
        param_lambda * s + param_alpha * g + param_beta * cp.power(g, 2) + param_mu * cp.abs(b)
    ) + param_gamma * cp.square(q[-1] - param_q_target)
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem


def _set_fixed_mpc_subproblem_params(problem, q_target, Q, B, G, mu, alpha, beta, lamb, gamma, efficiency, soc_loss):
    pd = problem.param_dict
    pd["Q"].value = Q
    pd["B"].value = B
    pd["alpha"].value = alpha
    pd["beta"].value = beta
    pd["lambda"].value = lamb
    pd["mu"].value = mu
    pd["gamma"].value = gamma
    pd["charge_efficiency"].value = efficiency
    pd["discharge_efficiency_inv"].value = 1 / efficiency  # assume discharge efficiency same as charge efficiency
    pd["soc_loss_per_hour"].value = soc_loss
    pd["q_target"].value = q_target
    pd["G"].value = G


def _set_mpc_subproblem_data_params(problem, t, h, l, R, q0):
    pd = problem.param_dict
    pd["l"].value = l[t : t + h]
    pd["R"].value = R[t : t + h]
    pd["q0"].value = q0


def run_mpc_perfect(
    l, R, G, Q, B, alpha, beta, lamb, gamma, mu, q_init, q_target, efficiency, soc_loss, H, delta=1, solver="CLARABEL"
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
        mu: battery degradation penalty
        q_target: target_soc at end of horizon
        gamma: penalty strength for deviating from end of horizon soc target
        efficiency: charge/discharge efficiency
        soc_loss: battery SOC loss rate per hour
        H: number of timesteps in horizon
        delta: hours per timestep
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

    mpc_subproblem = _make_mpc_subproblem(H, delta)
    _set_fixed_mpc_subproblem_params(
        problem=mpc_subproblem,
        q_target=q_target,
        Q=Q,
        B=B,
        G=G,
        mu=mu,
        alpha=alpha,
        beta=beta,
        lamb=lamb,
        gamma=gamma,
        efficiency=efficiency,
        soc_loss=soc_loss,
    )

    for t in range(T):
        h = min(H, T - t)

        # make a new subproblem to accomodate shorter horizon
        if h < H:
            mpc_subproblem = _make_mpc_subproblem(h, delta)
            _set_fixed_mpc_subproblem_params(
                problem=mpc_subproblem,
                q_target=q_target,
                Q=Q,
                B=B,
                G=G,
                mu=mu,
                alpha=alpha,
                beta=beta,
                lamb=lamb,
                gamma=gamma,
                efficiency=efficiency,
                soc_loss=soc_loss,
            )

        # update data and solve subproblem
        q0 = q_traj[t]
        _set_mpc_subproblem_data_params(problem=mpc_subproblem, t=t, h=h, l=l, R=R, q0=q0)
        mpc_subproblem.solve(solver=solver, warm_start=True)
        solution_valid = validate_solution_dynamics(mpc_subproblem, delta=delta)
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
