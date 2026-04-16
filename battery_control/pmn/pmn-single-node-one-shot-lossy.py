import marimo

__generated_with = "0.23.1"
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
    def process_single_node_data(G, add_event=False, verbose=False):
        df = pd.read_csv('../single_node_data.csv', index_col=0, parse_dates=True)
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
    def make_one_shot(l, R, G, delta=1):
        """
        make the prescient battery control problem

        args: 
            l: load
            R: available renewable generation
            G: max fossil generation
            delta: power to energy unit conversion constant
        
        return: 
            proeblem: the cvxpy problem object
        """
        T = R.shape[0]
        param_Q = cp.Parameter(nonneg=True, name='Q')
        param_alpha = cp.Parameter(nonneg=True, name='alpha')
        param_beta = cp.Parameter(nonneg=True, name='beta')
        param_lambda = cp.Parameter(nonneg=True, name='lambda')
        # param_mu = cp.Parameter(nonneg=True, name='mu')
        param_B = cp.Parameter(nonneg=True, name='B')
        param_y = cp.Parameter(T, nonneg=True, name='set_y')
        g = cp.Variable(T, nonneg=True, name='g') # fossil gen
        r = cp.Variable(T, nonneg=True, name='r') # non-dispatchable gen
        c = cp.Variable(T, nonneg=True, name='c') # curtailed non-dispatchablels
        b = cp.Variable(T, name='b') # battery power (output positive)
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
        objective = 1/T*cp.sum(param_lambda * s + param_alpha * g + param_beta * cp.power(g, 2) + 1e-10 * cp.sum(cp.abs(b)))
        problem = cp.Problem(cp.Minimize(objective), constraints)
        return problem

    return (make_one_shot,)


@app.cell
def _(mo):
    make_problem = mo.ui.run_button(label='make problem')
    add_abnormal_event = mo.ui.switch(label='add abnormal weather event')
    add_abnormal_event
    return (add_abnormal_event,)


@app.cell
def _(add_abnormal_event, process_single_node_data):
    G = 1
    l, R, shortfall, tidx, daily_df = process_single_node_data(G, add_abnormal_event.value, True)
    daily_df.plot(y=['load[MW]', 'pv[MW]', 'wind[MW]'])
    return G, R, l, shortfall, tidx


@app.cell
def _(l, mo):
    plot_start = mo.ui.slider(start=0, stop=len(l), label='plot start', full_width=True, value=13824)
    plot_length = mo.ui.slider(start=0, stop=len(l), step=1, label='plot length', value=26*24, full_width=True)
    show_batt_power_bounds = mo.ui.switch(label='show battery power bounds')
    show_cap_contrained = mo.ui.switch(label='show active capacity limits', value=True)
    mo.output.append(mo.hstack([plot_start,plot_length]))
    return plot_length, plot_start


@app.cell
def _(R, l, np, plot_length, plot_start, plt, shortfall, tidx):
    _fig, _ax = plt.subplots(nrows=3, sharex=True, figsize=(10, 5))
    _s = np.s_[int(plot_start.value):int(plot_start.value+plot_length.value)]
    _ax[0].plot(tidx[_s], l[_s])
    _ax[0].axhline(1, color='orange', ls=':', label='max fossil')
    _ax[0].legend()
    _ax[0].set_title('load')
    _ax[1].plot(tidx[_s], R[_s])
    _ax[1].set_title('renewables')
    _ax[2].plot(tidx[_s], shortfall[_s])
    _ax[2].set_title('shortfall')
    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    alpha_sldr = mo.ui.slider(start=0, stop=50, step=0.25, label='alpha', value=1.25, full_width=True)
    beta_sldr = mo.ui.slider(start=0, stop=50, step=0.25, label='beta', value=0.5, full_width=True)
    mu_sldr = mo.ui.slider(start=0, stop=50, step=0.25, label='mu', value=0, full_width=True)
    lambda_sldr = mo.ui.slider(start=0, stop=50, step=0.25, label='lambda', value=20.0, full_width=True)
    Q_sldr = mo.ui.number(start=0, stop=300, step=1, label='battery capacity [GWh]', value=4, full_width=True)
    bat_hours_sldr = mo.ui.number(start=0, stop=300, step=1, label='battery number of hours for full discharge', value=3, full_width=True)
    form = mo.md('''{alpha}\n{beta}\n{lambd}\n{mu}\n{Q}\n{bat_hours}''').batch(
        alpha=alpha_sldr,
        beta=beta_sldr,
        lambd=lambda_sldr,
        mu=mu_sldr,
        Q=Q_sldr,
        bat_hours=bat_hours_sldr
    )
    form
    return (form,)


@app.cell
def _(G, R, form, l, make_one_shot):
    problem = make_one_shot(l, R, G)
    problem.param_dict['alpha'].value = form.value['alpha']
    problem.param_dict['beta'].value = form.value['beta']
    # problem.param_dict['mu'].value = form.value['mu']
    problem.param_dict['lambda'].value = form.value['lambd']
    problem.param_dict['Q'].value = form.value['Q']
    problem.param_dict['B'].value = 1/form.value['bat_hours'] * form.value['Q']
    return (problem,)


@app.cell
def _(problem):
    problem.solve(solver='CLARABEL')
    return


@app.cell
def _(plt, problem):
    y = problem.var_dict['y'].value
    plt.hist(y)
    return


@app.cell
def _(cp, form, np, plot_length, plot_start, plt, problem, tidx):
    _s = np.s_[int(plot_start.value):int(plot_start.value+plot_length.value)]
    _fig, _ax = plt.subplots(nrows=5, sharex=True, figsize=(10, 6))
    _charged = np.isclose(problem.var_dict['q'].value[_s], form.value['Q'], atol=1e-2)
    _discharged = np.isclose(problem.var_dict['q'].value[_s], 0, atol=1e-2)
    _ax[0].plot(tidx[_s], problem.var_dict['q'].value[_s])
    _ax[0].axhline(0, color='red', ls='--', linewidth=0.5)
    _ax[0].axhline(form.value['Q'], color='red', ls='--', linewidth=0.5)
    _ax[0].axhline(0.5 * form.value['Q'], color='orange', ls=':', linewidth=0.5)
    _ax[0].set_title('battery SOC [GWh]')
    _ax[1].plot(tidx[_s], problem.var_dict['b'].value[_s])
    _ax[1].plot(tidx[_s], problem.var_dict['b_out'].value[_s], linewidth=0.5)
    _ax[1].plot(tidx[_s], -problem.var_dict['b_in'].value[_s], linewidth=0.5)
    _ax[1].axhline(0, color='orange', ls=':', linewidth=0.5)
    dumped_power = np.max([np.abs(problem.var_dict['b_out'].value[_s]), np.abs(problem.var_dict['b_in'].value[_s])], axis=0) - np.abs(problem.var_dict['b'].value[_s])
    _ax[1].set_title(f'battery power, dumped = {(1-0.98**2) * np.sum(dumped_power):.2f} GWh')
    _ax[2].plot(tidx[_s], problem.var_dict['g'].value[_s])
    _ax[2].set_ylim(-0.1, 1.1)
    utility_cost = cp.sum(problem.param_dict['alpha'] * problem.var_dict['g'][_s] + 
                   problem.param_dict['beta'] * cp.power(problem.var_dict['g'][_s], 2)).value
    _ax[2].set_title(f'utility power, cost = {utility_cost:.2f}')
    _ax[3].plot(tidx[_s], problem.var_dict['c'].value[_s])
    ax3_title = f"curtailed renewable power, total = {np.sum(problem.var_dict['c'].value[_s]):.2f}"
    _ax[4].plot(tidx[_s], problem.var_dict['s'].value[_s])
    ax4_title = f"curtailed load, total = {np.sum(problem.var_dict['s'].value[_s]):.2f}"
    _ax[3].set_title(ax3_title + ' GWh')
    _ax[4].set_title(ax4_title + ' GWh')
    plt.tight_layout()
    _fig
    return


@app.cell
def _(form, l, mo, np, problem):
    charged_times = np.isclose(problem.var_dict['q'].value, form.value['Q'], atol=1e-2)
    discharged_times = np.isclose(problem.var_dict['q'].value, 0, atol=1e-2)
    _vc = 1 / ((np.sum(charged_times)) / (len(l) / 24))
    _vd = 1 / ((np.sum(discharged_times)) / (len(l) / 24))
    # _va = (_vc + _vd) / 2 / 2
    _va = _va = 1 / ((np.sum(charged_times) + np.sum(discharged_times)) / (len(l) / 24))
    _asoc = np.average(problem.var_dict['q'].value) / form.value['Q']
    mo.md(f"""Average number of days between charged periods: {_vc:.2f}

    Average number of days between discharged periods: {_vd:.2f}

    Average number of days in decoupled problems: {_va:.2f}

    Average SoC: {_asoc:.2f}""")
    return


if __name__ == "__main__":
    app.run()
