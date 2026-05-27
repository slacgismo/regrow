import numpy as np


def battery_dynamics_contraints(q, q0, Q, b, b_out, b_in, B, charge_efficiency, discharge_efficiency, soc_loss, delta):
    constraints = [
        b == b_out - b_in,
        b_out <= B,
        b_in <= B,
        q[0] == q0,
        q <= Q,
        q[1:] == q[:-1] * (1 - soc_loss * delta) + delta * (charge_efficiency * b_in - b_out / discharge_efficiency),
    ]
    return constraints


def conservation_of_power_constraints(g, G, r, R, b, l, s, c):
    constraints = [g + r + b == l - s, g <= G, r <= R, s <= l, c == R - r]
    return constraints


def validate_solution_dynamics(problem, Q, B, efficiency, soc_loss, delta):
    vd = problem.var_dict
    valid = validate_battery_dynamics(
        q=vd["q"].value,
        b=vd["b"].value,
        b_out=vd["b_out"].value,
        b_in=vd["b_in"].value,
        Q=Q,
        B=B,
        q0=problem.param_dict["q0"].value,
        charge_efficiency=efficiency,
        discharge_efficiency=efficiency,
        soc_loss=soc_loss,
        delta=delta,
    )
    return valid


def validate_battery_dynamics(
    q, b, b_out, b_in, Q, B, q0, charge_efficiency, discharge_efficiency, soc_loss, delta, tol=1e-3
):
    violations = {
        "b = b_out - b_in": np.max(np.abs(b - (b_out - b_in))),
        "b_out <= B": np.max(b_out - B),
        "b_in <= B": np.max(b_in - B),
        "q[0] == q0": abs(q[0] - q0),
        "q <= Q": np.max(q - Q),
        "SOC dynamics": np.max(
            np.abs(
                q[1:]
                - (q[:-1] * (1 - soc_loss * delta) + delta * (charge_efficiency * b_in - b_out / discharge_efficiency))
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
