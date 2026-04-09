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

    return cp, mo, np, pd, plt


@app.cell
def _(np, pd):
    def process_single_node_data(G, add_event=False, verbose=False):
        df = pd.read_csv('single_node_data.csv', index_col=0, parse_dates=True)
        df[df.columns] = df.apply(pd.to_numeric, errors='coerce').fillna(0).astype('float64')
        df = df[~df.index.duplicated(keep='first')]
        df['load[MW]'] = df['load[MW]'].mask(df['load[MW]'] < 100).interpolate(limit_direction='both').ffill().bfill()
        # df.index += pd.Timedelta(hours=8)
        df = df.loc["2018":"2020"]
        if add_event:
            df.loc["2019-08-15":"2019-08-20", 'load[MW]'] *= 1.5
            df.loc["2019-08-15":"2019-08-20", 'pv[MW]'] *= 0.25
        daily_df = df.groupby(df.index.date).aggregate('sum') / 1000
        daily_df.index = pd.to_datetime(daily_df.index)

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
        return l, R, shortfall, df.index, daily_df

    return (process_single_node_data,)


@app.cell
def _(cp):
    def make_one_shot(l, R, G, delta=1, set_y=False):
        T = R.shape[0]
        param_Q = cp.Parameter(nonneg=True, name='Q')
        param_alpha = cp.Parameter(nonneg=True, name='alpha')
        param_beta = cp.Parameter(nonneg=True, name='beta')
        param_lambda = cp.Parameter(nonneg=True, name='lambda')
        param_B = cp.Parameter(nonneg=True, name='B')
        param_y = cp.Parameter(T, nonneg=True, name='set_y')
        g = cp.Variable(T, nonneg=True, name='g')
        r = cp.Variable(T, nonneg=True, name='r')
        c = cp.Variable(T, nonneg=True, name='c')
        b = cp.Variable(T, name='b')
        b_out = cp.Variable(T, nonneg=True, name='b_out')
        b_in = cp.Variable(T, nonneg=True, name='b_in')
        s = cp.Variable(T, nonneg=True, name='s')
        q = cp.Variable(T+1, nonneg=True, name='q')
        y = cp.Variable(T, nonneg=True, name='y')
        constraints = [
            g + r + b == l - s,
            0 <= g, g <= G,
            0 <= r, r <= R,
            0 <= s, s <= l,
            cp.abs(b) <= param_B,
            q <= param_Q,
            q[1:] == q[:-1] * (1-1e-6) + delta * (0.98* b_in - b_out/0.98),
            q[0] == 0.5*param_Q,
            b == b_out - b_in,
            b_out <= param_B * y,
            b_in <= param_B * (1 - y),
            y <= 1,
            c == R - r
        ]
        if set_y:
            constraints.append(y == param_y)
        objective = 1/T*cp.sum(param_lambda * s  + param_alpha * g + param_beta * cp.power(g, 2))
        problem = cp.Problem(cp.Minimize(objective), constraints)
        return problem

    return (make_one_shot,)


@app.cell
def _(mo):
    make_problem = mo.ui.run_button(label='make problem')
    run_study = mo.ui.run_button(label='run study')
    add_abnormal_event = mo.ui.switch(label='add abnormal weather event', value=True)
    solve_twice = mo.ui.switch(label='resolve to polish battery', value=False)
    mo.vstack([make_problem, run_study, add_abnormal_event, solve_twice])
    return add_abnormal_event, make_problem, run_study, solve_twice


@app.cell
def _(add_abnormal_event, process_single_node_data):
    G = 1
    l, R, shortfall, tidx, daily_df = process_single_node_data(G, add_abnormal_event.value, True)
    return G, R, l


@app.cell
def _(G, R, l, make_one_shot, make_problem, mo, solve_twice):
    mo.stop(not make_problem.value)

    if solve_twice.value:
        problem_zero = make_one_shot(l, R, G)
        problem = make_one_shot(l, R, G, set_y=True)
    else:
        problem = make_one_shot(l, R, G)
        problem_zero = None
    problem
    return (problem,)


@app.cell
def _(np):
    event = np.s_[14160:14160 + 8*24]
    metrics = [
        'load shedding',
        'fossil cost',
        'curtailed renewables',
        'total battery usage'
    ]
    return event, metrics


@app.cell
def _(cp, event, metrics, mo, np, pd, problem, run_study):
    mo.stop(not run_study.value)
    am_solving = True
    problem.param_dict['alpha'].value = 1.25
    problem.param_dict['beta'].value = 0.5
    problem.param_dict['Q'].value = 4
    problem.param_dict['B'].value = 0.333 * problem.param_dict['Q'].value # fully charge in 3 hours

    results_total = pd.DataFrame(columns=np.r_[['weight'], metrics])
    results_event = pd.DataFrame(columns=np.r_[['weight'], metrics])

    lambdas = np.logspace(-1.5, 1.5, 21)
    for _ix, _l in enumerate(mo.status.progress_bar(lambdas)):
        problem.param_dict['lambda'].value = _l
        problem.solve(verbose=False, solver='CLARABEL')
        results_total.loc[_ix] = [
            _l, 
            cp.sum(problem.var_dict['s']).value,
            cp.sum(problem.param_dict['alpha'] * problem.var_dict['g'] + 
                   problem.param_dict['beta'] * cp.power(problem.var_dict['g'], 2)).value,
            cp.sum(problem.var_dict['c']).value,
            cp.sum(cp.abs(problem.var_dict['b'])).value
        ]
        results_event.loc[_ix] = [
            _l, 
            cp.sum(problem.var_dict['s'][event]).value,
            cp.sum(problem.param_dict['alpha'] * problem.var_dict['g'][event] + 
                   problem.param_dict['beta'] * cp.power(problem.var_dict['g'][event], 2)).value,
            cp.sum(problem.var_dict['c'][event]).value,
            cp.sum(cp.abs(problem.var_dict['b'][event])).value
        ]
    return results_event, results_total


@app.cell
def _(plt, results_total):
    results_total.plot(x='load shedding', y='fossil cost', marker='.')
    plt.ylabel('fossil cost')
    # plt.xscale('log')
    # plt.yscale('log')
    plt.gcf()
    return


@app.cell
def _(metrics, plt, results_total):
    results_total.plot(x='weight', y=metrics[:2], marker='.', color=['C0', 'C1'])
    results_total.plot(x='weight', y=metrics[2:], marker='.', color=['C2', 'C5'], ax=plt.gca(), ls='--')
    plt.xscale('log')
    plt.xlabel('relative weight of two objectives')
    plt.gcf()
    return


@app.cell
def _(metrics, plt, results_event):
    results_event.plot(x='weight', y=metrics[:2], marker='.', color=['C0', 'C1'])
    results_event.plot(x='weight', y=metrics[2:], marker='.', color=['C2', 'C5'], ax=plt.gca(), ls='--')
    plt.xscale('log')
    plt.xlabel('relative weight of two objectives')
    plt.gcf()
    return


if __name__ == "__main__":
    app.run()
