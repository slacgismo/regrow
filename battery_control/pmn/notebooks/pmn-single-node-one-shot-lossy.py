import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import pathlib
    import sys

    import cvxpy as cp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    from controllers.one_shot import make_one_shot, load_one_shot_problem_data
    from controllers.constraints import validate_solution_dynamics
    from controllers.data_utils import process_single_node_data

    data_path = str(pathlib.Path(__file__).parent.parent.parent / 'single_node_data.csv')
    return (
        cp,
        data_path,
        load_one_shot_problem_data,
        make_one_shot,
        mo,
        np,
        pd,
        plt,
        process_single_node_data,
        validate_solution_dynamics,
    )


@app.cell
def _(mo):
    data_start_input = mo.ui.text(value='2019', label='data start')
    data_end_input = mo.ui.text(value='2019', label='data end')
    add_abnormal_event = mo.ui.switch(label='add abnormal weather event')
    event_start_input = mo.ui.text(value='2019-08-15', label='event start')
    event_end_input = mo.ui.text(value='2019-08-20', label='event end')
    event_load_factor_sldr = mo.ui.number(start=0.1, stop=5.0, step=0.05, label='event load factor', value=1.5, full_width=True)
    event_pv_factor_sldr = mo.ui.number(start=0.0, stop=1.0, step=0.05, label='event PV factor', value=0.25, full_width=True)
    mo.vstack([
        add_abnormal_event,
        mo.vstack([data_start_input, data_end_input, event_start_input, event_end_input, event_load_factor_sldr, event_pv_factor_sldr]),
    ])
    return (
        add_abnormal_event,
        data_end_input,
        data_start_input,
        event_end_input,
        event_load_factor_sldr,
        event_pv_factor_sldr,
        event_start_input,
    )


@app.cell
def _(
    add_abnormal_event,
    data_end_input,
    data_path,
    data_start_input,
    event_end_input,
    event_load_factor_sldr,
    event_pv_factor_sldr,
    event_start_input,
    process_single_node_data,
):
    G = 1
    l, R, shortfall, tidx, daily_df = process_single_node_data(
        G,
        data_path=data_path,
        data_start=data_start_input.value,
        data_end=data_end_input.value,
        add_event=add_abnormal_event.value,
        event_start=event_start_input.value,
        event_end=event_end_input.value,
        event_load_factor=event_load_factor_sldr.value,
        event_pv_factor=event_pv_factor_sldr.value,
        verbose=True,
    )
    daily_df.plot(y=['load[MW]', 'pv[MW]', 'wind[MW]'])
    return G, R, l, shortfall, tidx


@app.cell
def _(l, make_one_shot):
    T = len(l)
    one_shot_problem = make_one_shot(T, delta = 1)
    return (one_shot_problem,)


@app.cell
def _(mo):
    alpha_sldr = mo.ui.slider(start=0, stop=50, step=0.25, label='alpha', value=1.25, full_width=True)
    beta_sldr = mo.ui.slider(start=0, stop=50, step=0.25, label='beta', value=0.5, full_width=True)
    lambda_sldr = mo.ui.slider(start=0, stop=50, step=0.25, label='lambda', value=20.0, full_width=True)
    mu_exp_sldr = mo.ui.slider(start=-15, stop=2, step=0.5, label='mu (log base 10)', value=-5, full_width=True)
    Q_sldr = mo.ui.number(start=0, stop=300, step=1, label='battery capacity [GWh]', value=4, full_width=True)
    bat_hours_sldr = mo.ui.number(start=0, stop=300, step=1, label='battery number of hours for full discharge', value=3, full_width=True)
    power_efficiency_sldr = mo.ui.number(start=0, stop=1,label='power efficiency', value=0.98, full_width=True)
    soc_loss_sldr = mo.ui.number(start=0, stop=1,label='soc loss', value=1e-6, full_width=True)
    form = mo.md('''{alpha}\n{beta}\n{lambd}\n{mu_exp}\n{Q}\n{bat_hours}\n{power_efficiency}\n{soc_loss}''').batch(
        alpha=alpha_sldr,
        beta=beta_sldr,
        lambd=lambda_sldr,
        mu_exp=mu_exp_sldr,
        Q=Q_sldr,
        bat_hours=bat_hours_sldr,
        power_efficiency=power_efficiency_sldr,
        soc_loss=soc_loss_sldr
    )
    form
    return (form,)


@app.cell
def _(
    G,
    R,
    form,
    l,
    load_one_shot_problem_data,
    one_shot_problem,
    validate_solution_dynamics,
):
    load_one_shot_problem_data(one_shot_problem, l, R, G, form.value['Q']/2, form.value['Q'], 1/form.value['bat_hours'] * form.value['Q'], form.value['alpha'], form.value['beta'], form.value['lambd'], 10 ** form.value['mu_exp'], form.value['power_efficiency'], form.value['soc_loss'])
    one_shot_problem.solve(solver= 'clarabel')
    print(f"solution satisfies dynamics: {validate_solution_dynamics(one_shot_problem)}")
    return


@app.cell
def _(l, mo):
    plot_start = mo.ui.slider(start=0, stop=len(l), label='plot start', full_width=True, value=0)
    plot_length = mo.ui.slider(start=0, stop=len(l), step=1, label='plot length', value=24*10, full_width=True)
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
def _(cp, form, np, one_shot_problem, plot_length, plot_start, plt, tidx):
    _s = np.s_[int(plot_start.value):int(plot_start.value+plot_length.value)]
    _fig, _ax = plt.subplots(nrows=5, sharex=True, figsize=(10, 6))
    _ax[0].plot(tidx[_s], one_shot_problem.var_dict['q'].value[_s])
    _ax[0].axhline(0, color='red', ls='--', linewidth=0.5)
    _ax[0].axhline(form.value['Q'], color='red', ls='--', linewidth=0.5)
    _ax[0].axhline(0.5 * form.value['Q'], color='orange', ls=':', linewidth=0.5)
    _ax[0].set_title('battery SOC [GWh]')
    _ax[1].plot(tidx[_s], one_shot_problem.var_dict['b'].value[_s])
    _ax[1].plot(tidx[_s], one_shot_problem.var_dict['b_out'].value[_s], linewidth=0.5)
    _ax[1].plot(tidx[_s], -one_shot_problem.var_dict['b_in'].value[_s], linewidth=0.5)
    _ax[1].axhline(0, color='orange', ls=':', linewidth=0.5)
    dumped_power = np.max([np.abs(one_shot_problem.var_dict['b_out'].value[_s]), np.abs(one_shot_problem.var_dict['b_in'].value[_s])], axis=0) - np.abs(one_shot_problem.var_dict['b'].value[_s])
    _ax[1].set_title(f'battery power, dumped = {(1-0.98**2) * np.sum(dumped_power):.2f} GWh')
    _ax[2].plot(tidx[_s], one_shot_problem.var_dict['g'].value[_s])
    _ax[2].set_ylim(-0.1, 1.1)
    utility_cost = cp.sum(one_shot_problem.param_dict['alpha'] * one_shot_problem.var_dict['g'][_s] +
                   one_shot_problem.param_dict['beta'] * cp.power(one_shot_problem.var_dict['g'][_s], 2)).value
    _ax[2].set_title(f'utility power, cost = {utility_cost:.2f}')
    _ax[3].plot(tidx[_s], one_shot_problem.var_dict['c'].value[_s])
    ax3_title = f"curtailed renewable power, total = {np.sum(one_shot_problem.var_dict['c'].value[_s]):.2f}"
    _ax[4].plot(tidx[_s], one_shot_problem.var_dict['s'].value[_s])
    ax4_title = f"curtailed load, total = {np.sum(one_shot_problem.var_dict['s'].value[_s]):.2f}"
    _ax[3].set_title(ax3_title + ' GWh')
    _ax[4].set_title(ax4_title + ' GWh')
    plt.tight_layout()
    _fig
    return


@app.cell
def _(form, l, mo, np, one_shot_problem):
    charged_times = np.isclose(one_shot_problem.var_dict['q'].value, form.value['Q'], atol=1e-2)
    discharged_times = np.isclose(one_shot_problem.var_dict['q'].value, 0, atol=1e-2)
    _vc = 1 / ((np.sum(charged_times)) / (len(l) / 24))
    _vd = 1 / ((np.sum(discharged_times)) / (len(l) / 24))
    _va = 1 / ((np.sum(charged_times) + np.sum(discharged_times)) / (len(l) / 24))
    _asoc = np.average(one_shot_problem.var_dict['q'].value) / form.value['Q']
    mo.md(f"""Average number of days between charged periods: {_vc:.2f}

    Average number of days between discharged periods: {_vd:.2f}

    Average number of days in decoupled one_shot_problems: {_va:.2f}

    Average SoC: {_asoc:.2f}""")
    return charged_times, discharged_times


@app.cell
def _(charged_times, discharged_times, np):
    full = np.where(charged_times)[0]
    empty = np.where(discharged_times)[0]
    return


@app.cell
def _(one_shot_problem, pd, tidx):
    q = one_shot_problem.var_dict['q'].value
    b = one_shot_problem.var_dict['b'].value

    q_df = pd.Series(data = q[1:], index=tidx, name = "battery SOC")
    q_day = q_df.groupby(q_df.index.hour).agg(
        q10=lambda x: x.quantile(0.1),
        median="median",
        q90=lambda x: x.quantile(0.9))

    b_df = pd.Series(data = b, index=tidx, name = "battery power")
    b_day = b_df.groupby(b_df.index.hour).agg(
        q10=lambda x: x.quantile(0.1),
        median="median",
        q90=lambda x: x.quantile(0.9))
    return b_day, q_day


@app.cell
def _(b_day, form, plt, q_day):
    Q = form.value['Q']
    B = form.value['Q'] / form.value['bat_hours']
    plt.plot(q_day.index, q_day["median"])
    plt.fill_between(q_day.index, q_day["q10"], q_day["q90"], alpha=0.2)
    plt.axhline(Q, color='black', ls='--', linewidth=0.8, label='Q')
    plt.axhline(0, color='black', ls='--', linewidth=0.8)
    plt.legend()
    plt.title("daily median battery SOC")
    plt.show()
    plt.plot(b_day.index, b_day["median"])
    plt.fill_between(b_day.index, b_day["q10"], b_day["q90"], alpha=0.2)
    plt.axhline(B, color='black', ls='--', linewidth=0.8, label='B')
    plt.axhline(-B, color='black', ls='--', linewidth=0.8)
    plt.legend()
    plt.title("daily median battery power")
    return


if __name__ == "__main__":
    app.run()
