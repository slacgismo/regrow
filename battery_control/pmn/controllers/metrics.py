import cvxpy as cp
import numpy as np


def core_objective(s, g, b, lamb, alpha, beta, mu):
    return cp.mean(lamb * s + alpha * g + beta * cp.square(g) + mu * cp.abs(b))


def get_metrics_of_interest(s, g, b, c, lamb, alpha, beta, mu, delta=1, stress_mask=None):
    metrics = {"objective": core_objective(s, g, b, lamb, alpha, beta, mu).value}
    if stress_mask is None:
        n_days = len(s) * delta / 24
        metrics.update(
            {
                "load shedding / day": np.sum(s * delta) / n_days,
                "dispatched generation cost / day": np.sum((alpha * g + beta * g**2) * delta) / n_days,
                "battery throughput / day": np.sum(np.abs(b) * delta) / n_days,
                "curtailed non-dispatched generation / day": np.sum(c * delta) / n_days,
            }
        )
    else:
        normal_mask = ~stress_mask
        for prefix, mask in [("stress", stress_mask), ("normal", normal_mask)]:
            n_days = mask.sum() * delta / 24
            metrics.update(
                {
                    f"{prefix} objective": core_objective(s[mask], g[mask], b[mask], lamb, alpha, beta, mu).value,
                    f"{prefix} load shedding / day": np.sum(s[mask] * delta) / n_days,
                    f"{prefix} dispatched generation cost / day": np.sum((alpha * g[mask] + beta * g[mask] ** 2) * delta) / n_days,
                    f"{prefix} battery throughput / day": np.sum(np.abs(b[mask]) * delta) / n_days,
                    f"{prefix} curtailed non-dispatched generation / day": np.sum(c[mask] * delta) / n_days,
                }
            )
    return metrics
