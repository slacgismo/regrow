import cvxpy as cp
import numpy as np


def core_objective(s, g, b, lamb, alpha, beta, mu):
    return cp.mean(lamb * s + alpha * g + beta * cp.square(g) + mu * cp.abs(b))


def get_metrics_of_interest(s, g, b, c, lamb, alpha, beta, mu, delta=1):
    return {
        "objective": core_objective(s, g, b, lamb, alpha, beta, mu),
        "total load shedding": np.sum(s),
        "total dispatched generation cost": np.sum(alpha * g + beta * g**2),
        "total battery throughput": np.sum(np.abs(b) * delta),
        "total curtailed non-dispatched generation": np.sum(c),
    }
