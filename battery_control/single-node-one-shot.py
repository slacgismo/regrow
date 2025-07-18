import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import cvxpy as cp
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    return cp, mo, np, pd, plt


@app.cell
def _(np, pd):
    def process_single_node_data(G, verbose=False):
        df = pd.read_csv('single_node_data.csv', index_col=0, parse_dates=True)
        df[df.columns] = df.apply(pd.to_numeric, errors='coerce').fillna(0).astype('float64')
        df = df[~df.index.duplicated(keep='first')]
        df['load[MW]'] = df['load[MW]'].mask(df['load[MW]'] < 100).interpolate(limit_direction='both').ffill().bfill()
        df.index += pd.Timedelta(hours=8)
        df = df.loc["2018"]

        l = df['load[MW]'].to_numpy() / 1000
        s = df['pv[MW]'].to_numpy() / 1000
        w = df['wind[MW]'].to_numpy() / 1000
        R = w + s
        netload = l - (R + G)

        infeasible_indices = np.where(netload > 0)[0]
        shortfall = np.maximum(l - (R + G), 0)
        if verbose:
            print(f"percentage of shortfall times = {len(infeasible_indices) / len(l) * 100:.2f} %")
            print(f"average load = {np.mean(l):.2f} GW")
            print(f"average renewable generation = {np.mean(R):.2f} GW")
            print(f"average shortfall = {np.mean(shortfall):.2f} GW")
            print(f"maximum fossil generation = {np.max(G):.2f} GW")
        return l, R, shortfall, df.index
    return (process_single_node_data,)


@app.cell
def _(cp):
    def make_one_shot(l, R, G):
        T = R.shape[0]
        param_Q = cp.Parameter(nonneg=True, name='Q')
        param_alpha = cp.Parameter(nonneg=True, name='alpha')
        param_beta = cp.Parameter(nonneg=True, name='beta')
        param_gamma = cp.Parameter(nonneg=True, name='gamma')
        param_lambda = cp.Parameter(nonneg=True, name='lambda')
        B = cp.Variable(nonneg=True, name='B')
        b = cp.Variable(T, name='b')
        q = cp.Variable(T+1, nonneg=True, name='q')
        r = cp.Variable(T, nonneg=True, name='r')
        u = cp.Variable(T, nonneg=True, name='u')
        s = cp.Variable(T, nonneg=True, name='s')
        c = cp.Variable(T, nonneg=True, name='c')
        constraints = [
            c == R - r,
            r <= R,
            q <= param_Q,
            u <= G,
            cp.abs(b) <= B,
            cp.diff(q) == -b,
            s <= l,
            b + r + u == l - s,
            B == 0.33*param_Q,
            q[0] == 0.5*param_Q,
            q[-1] == 0.5*param_Q
        ]
        objective = 1/T*(param_gamma*cp.sum(c)+param_lambda*cp.sum(s)+param_alpha*cp.sum(u)+param_beta*cp.sum_squares(u))
        problem = cp.Problem(cp.Minimize(objective), constraints)
        return problem
    return (make_one_shot,)


@app.cell
def _(process_single_node_data):
    G = 1
    l, R, shortfall, tidx = process_single_node_data(G, True)
    return G, R, l, shortfall, tidx


@app.cell
def _(R, l, mo, plt, shortfall, tidx):
    _fig, _ax = plt.subplots(nrows=3, sharex=True, figsize=(10, 5))
    _ax[0].plot(tidx, l)
    _ax[1].plot(tidx, R)
    _ax[2].plot(tidx, shortfall)
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo):
    make_problem = mo.ui.run_button(label='make problem')
    make_problem
    return (make_problem,)


@app.cell
def _(G, R, l, make_one_shot, make_problem, mo):
    mo.stop(not make_problem.value)

    problem = make_one_shot(l, R, G)
    return (problem,)


@app.cell
def _(mo):
    alpha_sldr = mo.ui.slider(start=0, stop=50, step=0.25, label='alpha', value=1.25, full_width=True)
    beta_sldr = mo.ui.slider(start=0, stop=50, step=0.25, label='beta', value=0.5, full_width=True)
    gamma_sldr = mo.ui.slider(start=0, stop=50, step=0.25, label='gamma', value=5.0, full_width=True)
    lambda_sldr = mo.ui.slider(start=0, stop=50, step=0.25, label='lambda', value=20.0, full_width=True)
    Q_sldr = mo.ui.slider(start=0, stop=100, step=1, label='battery capacity [GWh]', value=4, full_width=True)
    return Q_sldr, alpha_sldr, beta_sldr, gamma_sldr, lambda_sldr


@app.cell
def _(Q_sldr, alpha_sldr, beta_sldr, gamma_sldr, lambda_sldr, mo):
    form = mo.md('''{alpha}\n{beta}\n{gamma}\n{lambd}\n{Q}\n''').batch(
        alpha=alpha_sldr,
        beta=beta_sldr,
        gamma=gamma_sldr,
        lambd=lambda_sldr,
        Q=Q_sldr
    )
    return (form,)


@app.cell
def _(form):
    form
    return


@app.cell
def _(mo):
    plot_start = mo.ui.slider(start=0, stop=365*24, label='plot start', full_width=True)
    plot_length = mo.ui.slider(start=0, stop=365*24+1, step=1, label='plot length', value=5*24, full_width=True)
    mo.hstack([plot_start,plot_length])
    return plot_length, plot_start


@app.cell
def _(am_solving, form, np, plot_length, plot_start, plt, problem):
    am_solving
    _s = np.s_[plot_start.value:plot_start.value+plot_length.value]
    _fig, _ax = plt.subplots(nrows=4, sharex=True, figsize=(10, 6))
    _ax[0].plot(problem.var_dict['q'].value[_s])
    _ax[0].axhline(0, color='red', ls='--')
    _ax[0].axhline(form.value['Q'], color='red', ls='--')
    _ax[0].axhline(0.5 * form.value['Q'], color='orange', ls=':')
    _ax[0].set_title('battery SOC')
    _ax[1].plot(problem.var_dict['b'].value[_s])
    _ax[1].axhline(0, color='red', ls='--')
    _ax[1].set_title('battery power')
    _ax[2].plot(problem.var_dict['u'].value[_s])
    _ax[2].set_ylim(-0.1, 1.1)
    _ax[2].set_title('utility power')
    _ax[3].plot(problem.var_dict['c'].value[_s])
    _ax[3].set_ylim(-0.1, 3.5)
    _ax[3].set_title('curtailed renewable power')
    plt.tight_layout()
    _fig
    return


@app.cell
def _(form, problem):
    problem.param_dict['alpha'].value = form.value['alpha']
    problem.param_dict['beta'].value = form.value['beta']
    problem.param_dict['gamma'].value = form.value['gamma']
    problem.param_dict['lambda'].value = form.value['lambd']
    problem.param_dict['Q'].value = form.value['Q']
    am_solving = True
    problem.solve(verbose=False, solver='CLARABEL')
    return (am_solving,)


@app.cell
def _(problem):
    problem.param_dict
    return


if __name__ == "__main__":
    app.run()
