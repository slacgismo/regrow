import numpy as np


def battery_dynamics_contraints(
    q, q0, Q, b, b_out, b_in, B, charge_efficiency, discharge_efficiency_inv, eta_storage, delta
):
    constraints = [
        b == b_out - b_in,
        b_out <= B,
        b_in <= B,
        q[0] == q0,
        q <= Q,
        q[1:]
        == q[:-1] * eta_storage**delta + delta * (charge_efficiency * b_in - discharge_efficiency_inv * b_out),
    ]
    return constraints


def conservation_of_power_constraints(g, G, r, R, b, l, s, c):
    constraints = [g + r + b == l - s, g <= G, r <= R, s <= l, c == R - r]
    return constraints


def validate_solution_dynamics(problem):
    vd = problem.var_dict
    return validate_battery_dynamics(b_out=vd["b_out"].value, b_in=vd["b_in"].value)


def validate_battery_dynamics(b_out, b_in, tol=1e-3):
    violation = np.max(np.minimum(b_in, b_out))
    if violation > tol:
        print(f"charge/discharge complementarity: {violation}")
        return False
    return True
