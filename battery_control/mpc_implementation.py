import marimo

__generated_with = "0.19.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import cvxpy as cp
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    return (cp,)


@app.cell
def _(G, R, cp, l):
    def make_one_shot(T, q0=None, delta=1, set_y=False):
        T = R.shape[0]
        param_Q = cp.Parameter(nonneg=True, name='Q')
        param_alpha = cp.Parameter(nonneg=True, name='alpha')
        param_beta = cp.Parameter(nonneg=True, name='beta')
        param_lambda = cp.Parameter(nonneg=True, name='lambda')
        param_mu = cp.Parameter(nonneg=True, name='mu')
        param_B = cp.Parameter(nonneg=True, name='B')
        param_y = cp.Parameter(T, nonneg=True, name='set_y')
        g = cp.Variable(T, nonneg=True, name='u')
        r = cp.Variable(T, nonneg=True, name='r')
        c = cp.Variable(T, nonneg=True, name='c')
        b = cp.Variable(T, name='b')
        b_out = cp.Variable(T, nonneg=True, name='b_out')
        b_in = cp.Variable(T, nonneg=True, name='b_in')
        s = cp.Variable(T, nonneg=True, name='s')
        q = cp.Variable(T+1, nonneg=True, name='q')
        y = cp.Variable(T, nonneg=True, name='y')
        if q0 is None:
            q0 = 0.5*param_Q
        constraints = [
            g + r + b == l - s,
            0 <= g, g <= G,
            0 <= r, r <= R,
            0 <= s, s <= l,
            cp.abs(b) <= param_B,
            q <= param_Q,
            q[1:] == q[:-1] * (1-1e-6) + delta * (0.98* b_in - b_out/0.98),
            q[0] == q0,
            b == b_out - b_in,
            b_out <= param_B * y,
            b_in <= param_B * (1 - y),
            y <= 1,
            c == R - r
        ]
        if set_y:
            constraints.append(y == param_y)
        objective = 1/T*cp.sum(param_lambda * s + param_mu * c + param_alpha * g + param_beta * cp.power(g, 2))
        problem = cp.Problem(cp.Minimize(objective), constraints)
        return problem

    return


if __name__ == "__main__":
    app.run()
