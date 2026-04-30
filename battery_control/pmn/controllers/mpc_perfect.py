import cvxpy as cp
import numpy as np


def _make_MPC_subproblem(H, delta):
    """make an MPC subproblem with horizon H"""

    #fixed problem parameters
    param_Q = cp.Parameter(nonneg=True, name='Q')
    param_B = cp.Parameter(nonneg=True, name='B')
    param_G = cp.Parameter(nonneg=True, name='G')
    param_alpha = cp.Parameter(nonneg=True, name='alpha')
    param_beta = cp.Parameter(nonneg=True, name='beta')
    param_lamb = cp.Parameter(nonneg=True, name='lamb')
    param_gamma = cp.Parameter(nonneg=True, name='gamma')
    param_efficiency = cp.Parameter(nonneg=True, name='power_efficiency')
    param_soc_loss = cp.Parameter(nonneg=True, name='soc_loss_per_hour')

    #problem data
    param_l = cp.Parameter(H, nonneg=True, name='l')
    param_R = cp.Parameter(H, nonneg=True, name='R')
    param_q0 = cp.Parameter(nonneg=True, name='q0')

    #variables
    g = cp.Variable(H, nonneg=True, name='g')
    r = cp.Variable(H, nonneg=True, name='r')
    c = cp.Variable(H, nonneg=True, name='c')
    b = cp.Variable(H, name='b')
    b_out = cp.Variable(H, nonneg=True, name='b_out')
    b_in = cp.Variable(H, nonneg=True, name='b_in')
    y = cp.Variable(H, nonneg=True, name='y')
    s = cp.Variable(H, nonneg=True, name='s')
    q = cp.Variable(H + 1, nonneg=True, name='q')

    constraints = [
        g + r + b == param_l - s,
        g <= param_G,
        r <= param_R,
        s <= param_l,
        cp.abs(b) <= param_B,
        q <= param_Q,
        q[1:] == q[:-1] * (1 - param_soc_loss * delta) + delta * (param_efficiency * b_in - b_out / param_efficiency),
        q[0] == param_q0,
        b == b_out - b_in,
        b_out <= param_B * y,
        b_in <= param_B * (1 - y),
        y <= 1,
        c == param_R - r,
    ]
    objective = (1 / H) * cp.sum(
        param_lamb * s + (param_alpha * g + param_beta * cp.power(g, 2)) + param_gamma * cp.abs(b)
    )
    return cp.Problem(cp.Minimize(objective), constraints)


def _set_fixed_problem_params(problem_params, Q, B, G, alpha, beta, lamb, gamma, power_efficiency, soc_loss):
    """
    sets the fixed battery and controller parameters
    """
    problem_params['Q'].value = Q
    problem_params['B'].value = B
    problem_params['G'].value = G
    problem_params['alpha'].value = alpha
    problem_params['beta'].value = beta
    problem_params['lamb'].value = lamb
    problem_params['gamma'].value = gamma
    problem_params['power_efficiency'].value = power_efficiency
    problem_params['soc_loss_per_hour'].value = soc_loss

def _set_data_params(problem_params, t, h, l, R, q0):
    """
    loads the problem data into problem params in the appropriate MPC slices

    args: 
        problem_params: the cvxpy problem param dict
        t: the time step an action must be implemented for
        l: load time series
        R: max available non-dispatchable generation time series
        q0: current battery state of charge 
    """
    problem_params['l'].value = l[t:t + h]
    problem_params['R'].value = R[t:t + h]
    problem_params['q0'].value = q0


def run_mpc_perfect(l, R, G, Q, B, alpha, beta, lamb, gamma, power_efficiency, soc_loss_per_hour, H, delta=1, solver='CLARABEL'):
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
        gamma: battery degradation penalty
        power_efficiency: charge/discharge efficiency
        soc_loss_per_hour: battery SOC loss rate per hour
        H: number of timesteps in horizon
        delta: timestep duration in hours
        solver: solver to use

    returns:
        dict of implemented variable trajectories
    """
    T = len(l)
    assert 1<= H <= T, f"horizon H={H} must be between 1 and T={T} (inclusive)"

    MPC_subproblem = _make_MPC_subproblem(H, G, delta)
    _set_fixed_problem_params(MPC_subproblem.param_dict, Q, B, alpha, beta, lamb, gamma, power_efficiency, soc_loss_per_hour)

    g_traj = np.zeros(T)
    r_traj = np.zeros(T)
    c_traj = np.zeros(T)
    b_traj = np.zeros(T)
    b_out_traj = np.zeros(T)
    b_in_traj = np.zeros(T)
    s_traj = np.zeros(T)
    y_traj = np.zeros(T)
    q_traj = np.zeros(T + 1)
    q_traj[0] = 0.5 * Q

    for t in range(T):
        h = min(H, T - t)

        # make a new subproblem to accomodate shorter horizon
        if h < H:
            MPC_subproblem = _make_MPC_subproblem(h, G, delta)
            _set_fixed_problem_params(MPC_subproblem.param_dict, Q, B, alpha, beta, lamb, gamma, power_efficiency, soc_loss_per_hour)

        #update data and solve subproblem
        q0 = q_traj[t]
        _set_data_params(MPC_subproblem.param_dict, t, h, l, R, q0)
        MPC_subproblem.solve(solver=solver, warm_start=True)

        #implement the actions prescribed for the immediate time step
        MPC_subproblem_solution = MPC_subproblem.var_dict
        g_traj[t] = MPC_subproblem_solution['g'].value[0]
        r_traj[t] = MPC_subproblem_solution['r'].value[0]
        c_traj[t] = MPC_subproblem_solution['c'].value[0]
        b_traj[t] = MPC_subproblem_solution['b'].value[0]
        b_out_traj[t] = MPC_subproblem_solution['b_out'].value[0]
        b_in_traj[t] = MPC_subproblem_solution['b_in'].value[0]
        s_traj[t] = MPC_subproblem_solution['s'].value[0]
        y_traj[t] = MPC_subproblem_solution['y'].value[0]
        q_traj[t + 1] = MPC_subproblem_solution['q'].value[1]

    return {
        'g': g_traj,
        'r': r_traj,
        'c': c_traj,
        'b': b_traj,
        'b_out': b_out_traj,
        'b_in': b_in_traj,
        's': s_traj,
        'y': y_traj,
        'q': q_traj,
    }
