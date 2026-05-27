import cvxpy as cp
import numpy as np


def core_objective(s, g, b, lamb, alpha, beta, mu):
    return cp.mean(lamb * s + alpha * g + beta * cp.square(g) + mu * cp.abs(b))


def get_metrics_of_interest(s, g, b, c, lamb, alpha, beta, mu, delta=1, stress_mask=None):
    metrics = {
        "objective": core_objective(s, g, b, lamb, alpha, beta, mu).value,
        "total load shedding": np.sum(s),
        "total dispatched generation cost": np.sum(alpha * g + beta * g**2),
        "total battery throughput": np.sum(np.abs(b) * delta),
        "total curtailed non-dispatched generation": np.sum(c),
    }
    if stress_mask is not None:
        normal_mask = ~stress_mask
        for prefix, mask in [("stress", stress_mask), ("normal", normal_mask)]:
            metrics.update(
                {
                    f"{prefix} objective": core_objective(s[mask], g[mask], b[mask], lamb, alpha, beta, mu).value,
                    f"{prefix} total load shedding": np.sum(s[mask]),
                    f"{prefix} total dispatched generation cost": np.sum(alpha * g[mask] + beta * g[mask] ** 2),
                    f"{prefix} total battery throughput": np.sum(np.abs(b[mask]) * delta),
                    f"{prefix} total curtailed non-dispatched generation": np.sum(c[mask]),
                }
            )
    return metrics
