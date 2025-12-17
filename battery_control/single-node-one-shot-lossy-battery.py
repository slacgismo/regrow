import marimo

__generated_with = "0.18.4"
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
        objective = 1/T*cp.sum(param_lambda * s + param_mu * c + param_alpha * g + param_beta * cp.power(g, 2))
        problem = cp.Problem(cp.Minimize(objective), constraints)
        return problem
    return (make_one_shot,)


@app.cell
def _(np):
    def naive_control_utility_priority(l, R, G, Q):
        T = R.shape[0]
        B = Q / 3
        b = np.zeros(T)
        q = np.zeros(T+1)
        q[0] = 0.5 * Q
        u = np.zeros(T)
        s = np.zeros(T)
        c = np.zeros(T)
        for _ix in range(T):
            net_load = l[_ix] - R[_ix]
            if net_load == 0:
                # exactly balences, everything else zero
                b[_ix] = 0
                # q[_ix+1] = q[_ix]
                u[_ix] = 0
                s[_ix] = 0
                c[_ix] = 0
            elif net_load < 0:
                # more renewable than load
                s[_ix] = 0
                u[_ix] = 0
                b[_ix] = max(max(net_load, -(Q - q[_ix])), -B)
                c[_ix] = b[_ix] - net_load
            else:
                # less renewable than load
                c[_ix] = 0
                u[_ix] = min(net_load, G)
                b[_ix] = min(min(net_load - u[_ix], q[_ix]), B)
                s[_ix] = net_load - u[_ix] - b[_ix]
            q[_ix+1] = q[_ix] - b[_ix]
        return_dict = {
            'q': q,
            'b': b,
            'u': u,
            'c': c,
            's': s
        }
        return return_dict
    return (naive_control_utility_priority,)


@app.cell
def _(np):
    def naive_control_battery_priority(l, R, G, Q):
        T = R.shape[0]
        B = Q / 3
        b = np.zeros(T)
        q = np.zeros(T+1)
        q[0] = 0.5 * Q
        u = np.zeros(T)
        s = np.zeros(T)
        c = np.zeros(T)
        for _ix in range(T):
            net_load = l[_ix] - R[_ix]
            if net_load == 0:
                # exactly balences, everything else zero
                b[_ix] = 0
                # q[_ix+1] = q[_ix]
                u[_ix] = 0
                s[_ix] = 0
                c[_ix] = 0
            elif net_load < 0:
                # more renewable than load
                s[_ix] = 0
                u[_ix] = 0
                b[_ix] = max(max(net_load, -(Q - q[_ix])), -B)
                c[_ix] = b[_ix] - net_load
            else:
                # less renewable than load
                c[_ix] = 0
                b[_ix] = min(min(net_load, q[_ix]), B)
                u[_ix] = min(net_load - b[_ix], G)
                s[_ix] = net_load - u[_ix] - b[_ix]
            q[_ix+1] = q[_ix] - b[_ix]
        return_dict = {
            'q': q,
            'b': b,
            'u': u,
            'c': c,
            's': s
        }
        return return_dict
    return (naive_control_battery_priority,)


@app.cell
def _(add_abnormal_event, process_single_node_data):
    G = 1
    l, R, shortfall, tidx, daily_df = process_single_node_data(G, add_abnormal_event.value, True)
    return G, R, daily_df, l, shortfall, tidx


@app.cell
def _(daily_df, mo, plt):
    daily_df.plot(y=['load[MW]', 'pv[MW]', 'wind[MW]'])
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(daily_df, plt):
    daily_df.loc["2019-08-12":"2019-08-23"].plot(y=['load[MW]', 'pv[MW]', 'wind[MW]'], marker='.')
    plt.legend(['load[GWh]', 'pv[GWh]', 'wind[GWh]'])
    plt.gcf()
    return


@app.cell
def _(mo):
    make_problem = mo.ui.run_button(label='make problem')
    add_abnormal_event = mo.ui.switch(label='add abnormal weather event')
    mo.vstack([make_problem, add_abnormal_event])
    return add_abnormal_event, make_problem


@app.cell
def _(G, R, l, make_one_shot, make_problem, mo):
    mo.stop(not make_problem.value)

    problem_zero = make_one_shot(l, R, G)
    problem = make_one_shot(l, R, G, set_y=True)
    # naive_problem = make_naive_control(l, R, G)
    return problem, problem_zero


@app.cell
def _(mo):
    alpha_sldr = mo.ui.slider(start=0, stop=50, step=0.25, label='alpha', value=1.25, full_width=True)
    beta_sldr = mo.ui.slider(start=0, stop=50, step=0.25, label='beta', value=0.5, full_width=True)
    mu_sldr = mo.ui.slider(start=0, stop=50, step=0.25, label='mu', value=5.0, full_width=True)
    lambda_sldr = mo.ui.slider(start=0, stop=50, step=0.25, label='lambda', value=20.0, full_width=True)
    Q_sldr = mo.ui.number(start=0, stop=300, step=1, label='battery capacity [GWh]', value=4, full_width=True)
    return Q_sldr, alpha_sldr, beta_sldr, lambda_sldr, mu_sldr


@app.cell
def _(Q_sldr, alpha_sldr, beta_sldr, lambda_sldr, mo, mu_sldr):
    form = mo.md('''{alpha}\n{beta}\n{lambd}\n{mu}\n{Q}\n''').batch(
        alpha=alpha_sldr,
        beta=beta_sldr,
        lambd=lambda_sldr,
        mu=mu_sldr,
        Q=Q_sldr
    )
    return (form,)


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
    # mo.mpl.interactive(_fig)
    _fig
    return


@app.cell
def _(l, mo):
    ### original view
    # plot_start = mo.ui.slider(start=0, stop=len(l), label='plot start', full_width=True, value=14069)
    # plot_length = mo.ui.slider(start=0, stop=len(l), step=1, label='plot length', value=12*24, full_width=True)
    ### tight vew
    # plot_start = mo.ui.slider(start=0, stop=len(l), label='plot start', full_width=True, value=14069+24*4.3)
    # plot_length = mo.ui.slider(start=0, stop=len(l), step=1, label='plot length', value=3.5*24, full_width=True)
    ## wide view
    plot_start = mo.ui.slider(start=0, stop=len(l), label='plot start', full_width=True, value=13824)
    plot_length = mo.ui.slider(start=0, stop=len(l), step=1, label='plot length', value=26*24, full_width=True)
    show_batt_power_bounds = mo.ui.switch(label='show battery power bounds')
    show_cap_contrained = mo.ui.switch(label='show active capacity limits', value=True)
    show_battery_priority = mo.ui.switch(label='show battery priority controller')
    show_utility_priority = mo.ui.switch(label='show utility priority controller')
    mo.output.append(mo.hstack([plot_start,plot_length]))
    mo.output.append(mo.hstack([show_batt_power_bounds, show_cap_contrained, show_battery_priority, show_utility_priority]))
    return (
        plot_length,
        plot_start,
        show_batt_power_bounds,
        show_battery_priority,
        show_cap_contrained,
        show_utility_priority,
    )


@app.cell
def _(
    am_solving,
    form,
    naive_bp,
    naive_up,
    np,
    plot_length,
    plot_start,
    plt,
    problem,
    show_batt_power_bounds,
    show_battery_priority,
    show_cap_contrained,
    show_utility_priority,
    tidx,
):
    am_solving
    _s = np.s_[int(plot_start.value):int(plot_start.value+plot_length.value)]
    _fig, _ax = plt.subplots(nrows=5, sharex=True, figsize=(10, 6))
    _charged = np.isclose(problem.var_dict['q'].value[_s], form.value['Q'], atol=1e-2)
    _discharged = np.isclose(problem.var_dict['q'].value[_s], 0, atol=1e-2)
    _ax[0].plot(tidx[_s], problem.var_dict['q'].value[_s])
    if show_cap_contrained.value:
        _ax[0].plot(tidx[_s][_charged], problem.var_dict['q'].value[_s][_charged], ls='none', marker='.', color='blue')
        _ax[0].plot(tidx[_s][_discharged], problem.var_dict['q'].value[_s][_discharged], ls='none', marker='.', color='orange')
    _ax[0].axhline(0, color='red', ls='--', linewidth=0.5)
    _ax[0].axhline(form.value['Q'], color='red', ls='--', linewidth=0.5)
    _ax[0].axhline(0.5 * form.value['Q'], color='orange', ls=':', linewidth=0.5)
    _ax[0].set_title('battery SOC')
    _ax[1].plot(tidx[_s], problem.var_dict['b'].value[_s])
    _ax[1].plot(tidx[_s], problem.var_dict['b_out'].value[_s], linewidth=0.5)
    _ax[1].plot(tidx[_s], -problem.var_dict['b_in'].value[_s], linewidth=0.5)
    # _ax[1].plot(tidx[_s], 2*problem.var_dict['y'].value[_s] - 1, linewidth=0.5)
    if show_batt_power_bounds.value:
        _ax[1].axhline(problem.var_dict['B'].value, color='red', ls='--', linewidth=0.5)
        _ax[1].axhline(-problem.var_dict['B'].value, color='red', ls='--', linewidth=0.5)
    _ax[1].axhline(0, color='orange', ls=':', linewidth=0.5)
    _ax[1].set_title('battery power')
    _ax[2].plot(tidx[_s], problem.var_dict['u'].value[_s])
    if show_cap_contrained.value:
        _ax[2].plot(tidx[_s][_charged], problem.var_dict['u'].value[_s][_charged], ls='none', marker='.', color='blue')
        _ax[2].plot(tidx[_s][_discharged], problem.var_dict['u'].value[_s][_discharged], ls='none', marker='.', color='orange')
    _ax[2].set_ylim(-0.1, 1.1)
    _ax[2].set_title('utility power')
    _ax[3].plot(tidx[_s], problem.var_dict['c'].value[_s])
    if show_cap_contrained.value:
        _ax[3].plot(tidx[_s][_charged], problem.var_dict['c'].value[_s][_charged], ls='none', marker='.', color='blue')
        _ax[3].plot(tidx[_s][_discharged], problem.var_dict['c'].value[_s][_discharged], ls='none', marker='.', color='orange')
    # _ax[3].set_ylim(-0.1 * np.max(problem.var_dict['c'].value), 1.1*np.max(problem.var_dict['c'].value))
    ax3_title = f'curtailed renewable power, total = {np.sum(problem.var_dict['c'].value[_s]):.2f}'
    _ax[4].plot(tidx[_s], problem.var_dict['s'].value[_s])
    if show_cap_contrained.value:
        _ax[4].plot(tidx[_s][_charged], problem.var_dict['s'].value[_s][_charged], ls='none', marker='.', color='blue')
        _ax[4].plot(tidx[_s][_discharged], problem.var_dict['s'].value[_s][_discharged], ls='none', marker='.', color='orange')
    # _ax[4].set_ylim(-0.1 * np.max(problem.var_dict['s'].value), 1.1*np.max(problem.var_dict['s'].value))
    ax4_title = f'curtailed load, total = {np.sum(problem.var_dict['s'].value[_s]):.2f}'
    if show_battery_priority.value:
        _ax[0].plot(tidx[_s], naive_bp['q'][_s], linewidth=0.75)
        _ax[1].plot(tidx[_s], naive_bp['b'][_s], linewidth=0.75)
        _ax[2].plot(tidx[_s], naive_bp['u'][_s], linewidth=0.75)
        _ax[3].plot(tidx[_s], naive_bp['c'][_s], linewidth=0.75)
        _ax[4].plot(tidx[_s], naive_bp['s'][_s], linewidth=0.75)
        ax3_title += f', {np.sum(naive_bp['c'][_s]):.2f}'
        ax4_title += f', {np.sum(naive_bp['s'][_s]):.2f}'
    if show_utility_priority.value:
        _ax[0].plot(tidx[_s], naive_up['q'][_s], linewidth=0.75)
        _ax[1].plot(tidx[_s], naive_up['b'][_s], linewidth=0.75)
        _ax[2].plot(tidx[_s], naive_up['u'][_s], linewidth=0.75)
        _ax[3].plot(tidx[_s], naive_up['c'][_s], linewidth=0.75)
        _ax[4].plot(tidx[_s], naive_up['s'][_s], linewidth=0.75)
        ax3_title += f', {np.sum(naive_up['c'][_s]):.2f}'
        ax4_title += f', {np.sum(naive_up['s'][_s]):.2f}'
    _ax[3].set_title(ax3_title + ' GWh')
    _ax[4].set_title(ax4_title + ' GWh')
    plt.tight_layout()
    _fig
    return


@app.cell
def _(am_solving, form, l, mo, np, problem):
    am_solving
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


@app.cell
def _(form):
    form
    return


@app.cell
def _(
    G,
    R,
    form,
    l,
    naive_control_battery_priority,
    naive_control_utility_priority,
    np,
    problem,
    problem_zero,
):
    problem_zero.param_dict['alpha'].value = form.value['alpha']
    problem_zero.param_dict['beta'].value = form.value['beta']
    problem_zero.param_dict['mu'].value = form.value['mu']
    problem_zero.param_dict['lambda'].value = form.value['lambd']
    problem_zero.param_dict['Q'].value = form.value['Q']
    problem_zero.param_dict['B'].value = 0.333 * form.value['Q'] # fully charge in 3 hours
    problem_zero.solve(verbose=False, solver='CLARABEL')
    problem.param_dict['alpha'].value = form.value['alpha']
    problem.param_dict['beta'].value = form.value['beta']
    problem.param_dict['mu'].value = form.value['mu']
    problem.param_dict['lambda'].value = form.value['lambd']
    problem.param_dict['Q'].value = form.value['Q']
    problem.param_dict['B'].value = 0.333 * form.value['Q'] # fully charge in 3 hours
    problem.param_dict['set_y'].value = np.round(problem_zero.var_dict['y'].value)
    am_solving = True
    problem.solve(verbose=False, solver='CLARABEL')
    naive_up = naive_control_utility_priority(l, R, G, form.value['Q'])
    naive_bp = naive_control_battery_priority(l, R, G, form.value['Q'])
    return am_solving, naive_bp, naive_up


@app.cell
def _(problem):
    problem.param_dict
    return


if __name__ == "__main__":
    app.run()
