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
    from controllers.mpc_perfect import run_mpc_perfect
    from controllers.constraints import validate_solution_dynamics
    from controllers.data_utils import process_single_node_data
    from plot_utils import plot_solution

    data_path = str(pathlib.Path(__file__).parent.parent.parent / "single_node_data.csv")
    return (
        data_path,
        load_one_shot_problem_data,
        make_one_shot,
        mo,
        np,
        pd,
        plot_solution,
        plt,
        process_single_node_data,
        run_mpc_perfect,
        validate_solution_dynamics,
    )


@app.cell(hide_code=True)
def _(mo):
    data_start_input = mo.ui.text(value="2019", label="data start")
    data_end_input = mo.ui.text(value="2019", label="data end")
    add_abnormal_event = mo.ui.switch(label="add abnormal weather event")
    event_start_input = mo.ui.text(value="2019-08-15", label="event start")
    event_end_input = mo.ui.text(value="2019-08-20", label="event end")
    event_load_factor_sldr = mo.ui.number(
        start=0.1, stop=5.0, step=0.05, label="event load factor", value=1.5, full_width=True
    )
    event_pv_factor_sldr = mo.ui.number(
        start=0.0, stop=1.0, step=0.05, label="event PV factor", value=0.25, full_width=True
    )
    mo.vstack(
        [
            add_abnormal_event,
            mo.vstack(
                [
                    data_start_input,
                    data_end_input,
                    event_start_input,
                    event_end_input,
                    event_load_factor_sldr,
                    event_pv_factor_sldr,
                ]
            ),
        ]
    )
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
    daily_df.plot(y=["load[MW]", "pv[MW]", "wind[MW]"])
    return G, R, l, tidx


@app.cell(hide_code=True)
def _(mo):
    alpha_sldr = mo.ui.slider(start=0, stop=50, step=0.25, label="alpha", value=1.25, full_width=True)
    beta_sldr = mo.ui.slider(start=0, stop=50, step=0.25, label="beta", value=0.5, full_width=True)
    lambda_sldr = mo.ui.slider(start=0, stop=50, step=0.25, label="lambda", value=20.0, full_width=True)
    mu_exp_sldr = mo.ui.slider(start=-15, stop=2, step=0.5, label="mu (log base 10)", value=-3, full_width=True)
    Q_sldr = mo.ui.number(start=0, stop=300, step=1, label="battery capacity [GWh]", value=4, full_width=True)
    bat_hours_sldr = mo.ui.number(
        start=0, stop=300, step=1, label="battery number of hours for full discharge", value=3, full_width=True
    )
    power_efficiency_sldr = mo.ui.number(start=0, stop=1, label="power efficiency", value=0.98, full_width=True)
    soc_loss_sldr = mo.ui.number(start=0, stop=1, label="soc loss", value=1e-6, full_width=True)
    form = mo.md("""{alpha}\n{beta}\n{lambd}\n{mu_exp}\n{Q}\n{bat_hours}\n{power_efficiency}\n{soc_loss}""").batch(
        alpha=alpha_sldr,
        beta=beta_sldr,
        lambd=lambda_sldr,
        mu_exp=mu_exp_sldr,
        Q=Q_sldr,
        bat_hours=bat_hours_sldr,
        power_efficiency=power_efficiency_sldr,
        soc_loss=soc_loss_sldr,
    )
    form
    return (form,)


@app.cell
def _(l, make_one_shot):
    T = len(l)
    one_shot_problem = make_one_shot(T, delta=1)
    return (one_shot_problem,)


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
    form
    load_one_shot_problem_data(
        one_shot_problem,
        l,
        R,
        G,
        form.value["Q"] / 2,
        form.value["Q"],
        1 / form.value["bat_hours"] * form.value["Q"],
        form.value["alpha"],
        form.value["beta"],
        form.value["lambd"],
        10 ** form.value["mu_exp"],
        form.value["power_efficiency"],
        form.value["soc_loss"],
    )
    one_shot_problem.solve(solver="clarabel")
    print(
        f"solution satisfies dynamics: {validate_solution_dynamics(problem=one_shot_problem, B=1/form.value['bat_hours'] * form.value['Q'], Q=form.value['Q'], efficiency=form.value['power_efficiency'], soc_loss=form.value['soc_loss'])}"
    )
    one_shot_solution = {var:one_shot_problem.var_dict[var].value for var in one_shot_problem.var_dict.keys()}
    return (one_shot_solution,)


@app.cell(hide_code=True)
def _(l, mo):
    H_sldr = mo.ui.number(start=1, stop=len(l), step=1, label="MPC horizon H (in time steps)", value=24, full_width=True)
    q_target_frac_sldr = mo.ui.slider(start=0.0, stop=1.0, step=0.05, label="q_target (fraction of Q)", value=1.0, full_width=True)
    gamma_sldr = mo.ui.slider(start=0.0, stop=10.0, step=0.1, label="gamma", value=0.0, full_width=True)
    form_mpc = mo.md("""{H}\n{q_target_frac}\n{gamma}""").batch(
        H=H_sldr,
        q_target_frac=q_target_frac_sldr,
        gamma=gamma_sldr,
    )
    form_mpc
    return (form_mpc,)


@app.cell
def _(G, R, form, form_mpc, l, run_mpc_perfect):
    mpc_solution = run_mpc_perfect(
        l=l,
        R=R,
        G=G,
        Q=form.value["Q"],
        B=1 / form.value["bat_hours"] * form.value["Q"],
        alpha=form.value["alpha"],
        beta=form.value["beta"],
        lamb=form.value["lambd"],
        gamma=form_mpc.value["gamma"],
        mu=10 ** form.value["mu_exp"],
        q_init=form.value["Q"] / 2,
        q_target=form_mpc.value["q_target_frac"] * form.value["Q"],
        efficiency=form.value["power_efficiency"],
        soc_loss=form.value["soc_loss"],
        H=form_mpc.value["H"],
    )
    return (mpc_solution,)


@app.cell
def _(add_abnormal_event, event_end_input, event_start_input, l, mo, pd, tidx):
    if add_abnormal_event.value:
        _event_start = int(tidx.searchsorted(pd.Timestamp(event_start_input.value)))
        _event_end = int(tidx.searchsorted(pd.Timestamp(event_end_input.value)))
        _event_duration = _event_end - _event_start
        _default_length = 2 * _event_duration
        _default_start = max(0, _event_start - _event_duration // 2)
    else:
        _default_start = 0
        _default_length = 24 * 7
    plot_start = mo.ui.slider(start=0, stop=len(l), label="plot start", full_width=True, value=_default_start)
    plot_length = mo.ui.slider(start=0, stop=len(l), step=1, label="plot length", value=_default_length, full_width=True)
    mo.output.append(mo.hstack([plot_start, plot_length]))
    return plot_length, plot_start


@app.cell
def _(
    form,
    np,
    one_shot_solution,
    plot_length,
    plot_solution,
    plot_start,
    plt,
    tidx,
):
    s = np.s_[int(plot_start.value) : int(plot_start.value + plot_length.value)]
    fig_one_shot = plot_solution(
        one_shot_solution,
        tidx=tidx,
        s=s,
        Q=form.value["Q"],
        B=form.value["Q"] / form.value["bat_hours"],
        alpha=form.value["alpha"],
        beta=form.value["beta"],
        efficiency=form.value["power_efficiency"],
    )
    plt.show()
    return (s,)


@app.cell
def _(form, mpc_solution, plot_solution, plt, s, tidx):
    fig_mpc = plot_solution(
        mpc_solution,
        tidx=tidx,
        s=s,
        Q=form.value["Q"],
        B=form.value["Q"] / form.value["bat_hours"],
        alpha=form.value["alpha"],
        beta=form.value["beta"],
        efficiency=form.value["power_efficiency"],
    )
    plt.show()

    return


if __name__ == "__main__":
    app.run()
