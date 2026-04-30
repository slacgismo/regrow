import cvxpy as cp
import numpy as np


def battery_dynamics_contraints(
    q, q0, Q, b, b_out, b_in, y, B, charge_efficiency, dishcarge_efficiency_inv, soc_loss, delta
):
    constraints = [
        b == b_out - b_in,
        cp.abs(b) <= B,
        b_out <= B * y,
        b_in <= B * (1 - y),
        y <= 1,
        q[0] == q0,
        q <= Q,
        q[1:]
        == q[:-1] * (1 - soc_loss * delta) + delta * (charge_efficiency * b_in - dishcarge_efficiency_inv * b_out),
    ]
    return constraints


def conservation_of_power_constraints(g, G, r, R, b, l, s, c):
    constraints = [g + r + b == l - s, g <= G, r <= R, s <= l, c == R - r]
    return constraints


def validate_solution_dynamics(problem, delta=1):
    vd = problem.var_dict
    pd = problem.param_dict
    valid = validate_battery_dynamics(
        q=vd["q"].value,
        b=vd["b"].value,
        b_out=vd["b_out"].value,
        b_in=vd["b_in"].value,
        y=vd["y"].value,
        Q=pd["Q"].value,
        B=pd["B"].value,
        q0=pd["q0"].value,
        charge_efficiency=pd["charge_efficiency"].value,
        discharge_efficiency_inv=pd["discharge_efficiency_inv"].value,
        soc_loss=pd["soc_loss_per_hour"].value,
        delta=delta,
    )
    return valid


def validate_battery_dynamics(
    q, b, b_out, b_in, y, Q, B, q0, charge_efficiency, discharge_efficiency_inv, soc_loss, delta, tol=1e-3
):
    violations = {
        "b = b_out - b_in": np.max(np.abs(b - (b_out - b_in))),
        "|b| <= B": np.max(np.abs(b) - B),
        "b_out <= B*y": np.max(b_out - B * y),
        "b_in <= B*(1-y)": np.max(b_in - B * (1 - y)),
        "y <= 1": np.max(y - 1),
        "q[0] == q0": abs(q[0] - q0),
        "q <= Q": np.max(q - Q),
        "SOC dynamics": np.max(
            np.abs(
                q[1:]
                - (
                    q[:-1] * (1 - soc_loss * delta)
                    + delta * (charge_efficiency * b_in - discharge_efficiency_inv * b_out)
                )
            )
        ),
        "charge/discharge complementarity": np.max(np.minimum(b_in, b_out)),
    }
    violated = {k: v for k, v in violations.items() if v > tol}
    if violated:
        for k, v in violated.items():
            print(f"{k}: {v}")
        return False
    return True
