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
        # df.index += pd.Timedelta(hours=8)
        df = df.loc["2018":"2019"]

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
        Q = cp.Variable(nonneg=True, name='Q')
        B = cp.Variable(nonneg=True, name='B')
        b = cp.Variable(T, name='b')
        q = cp.Variable(T+1, nonneg=True, name='q')
        r = cp.Variable(T, nonneg=True, name='r')
        u = cp.Variable(T, nonneg=True, name='u')
        s = cp.Variable(T, nonneg=True, name='s')
        c = cp.Variable(T, nonneg=True, name='c')
        u_bar = cp.Variable(nonneg=True, name='u_bar')
        constraints = [
            c == R - r,
            # c == 0,
            r <= R,
            s == 0,
            q <= Q,
            u <= G,
            u == u_bar,
            cp.abs(b) <= B,
            cp.diff(q) == -b,
            s <= l,
            b + r + u == l - s,
            B == 0.33*Q,
            q[0] == 1*Q,
            q[-1] == 1*Q
        ]
        objective = Q + 1000*u_bar
        problem = cp.Problem(cp.Minimize(objective), constraints)
        return problem
    return (make_one_shot,)


@app.cell
def _(process_single_node_data):
    G = 1
    l, R, shortfall, tidx = process_single_node_data(G, True)
    return G, R, l, shortfall, tidx


@app.cell
def _(mo):
    make_problem = mo.ui.run_button(label='make problem')
    make_problem
    return (make_problem,)


@app.cell
def _(G, R, l, make_one_shot, make_problem, mo):
    mo.stop(not make_problem.value)

    problem = make_one_shot(l, R, G)
    am_solving = True
    problem.solve(verbose=False, solver='CLARABEL')
    problem.var_dict['Q'].value
    return am_solving, problem


@app.cell
def _(R, l, np, plot_length, plot_start, plt, shortfall, tidx):
    _fig, _ax = plt.subplots(nrows=3, sharex=True, figsize=(10, 5))
    _s = np.s_[plot_start.value:plot_start.value+plot_length.value]
    _ax[0].plot(tidx[_s], l[_s])
    _ax[0].set_title('load')
    _ax[1].plot(tidx[_s], R[_s])
    _ax[1].set_title('renewables')
    _ax[2].plot(tidx[_s], shortfall[_s])
    _ax[2].set_title('shortfall')
    plt.tight_layout()
    # mo.mpl.interactive(_fig)
    _fig
    return


@app.cell
def _(l, mo):
    plot_start = mo.ui.slider(start=0, stop=len(l), label='plot start', full_width=True)
    plot_length = mo.ui.slider(start=0, stop=len(l), step=1, label='plot length', value=5*24, full_width=True)
    mo.hstack([plot_start,plot_length])
    return plot_length, plot_start


@app.cell
def _(am_solving, np, plot_length, plot_start, plt, problem):
    am_solving
    _s = np.s_[plot_start.value:plot_start.value+plot_length.value]
    _fig, _ax = plt.subplots(nrows=5, sharex=True, figsize=(10, 6))
    _ax[0].plot(problem.var_dict['q'].value[_s])
    _ax[0].axhline(0, color='red', ls='--')
    _ax[0].axhline(problem.var_dict['Q'].value, color='red', ls='--')
    _ax[0].axhline(0.5 * problem.var_dict['Q'].value, color='orange', ls=':')
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
    _ax[4].plot(problem.var_dict['s'].value[_s])
    _ax[4].set_ylim(-0.05, 0.6)
    _ax[4].set_title('curtailed load')
    plt.tight_layout()
    _fig
    return


@app.cell
def _():
    return


@app.cell
def _(problem):
    problem.param_dict
    return


if __name__ == "__main__":
    app.run()
