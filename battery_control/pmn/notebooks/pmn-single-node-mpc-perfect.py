import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import pathlib
    import sys

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    from controllers.one_shot import make_one_shot, load_one_shot_problem_data
    from controllers.mpc_perfect import run_mpc_perfect
    from controllers.constraints import validate_solution_dynamics
    from controllers.metrics import get_metrics_of_interest
    from controllers.data_utils import process_single_node_data
    from plot_utils import compute_partition, plot_solution, plot_timeseries, solution_diff_heatmaps, solution_heatmaps

    data_path = str(pathlib.Path(__file__).parent.parent.parent / "single_node_data.csv")
    return (
        compute_partition,
        data_path,
        get_metrics_of_interest,
        load_one_shot_problem_data,
        make_one_shot,
        mo,
        np,
        pd,
        plot_solution,
        plot_timeseries,
        plt,
        process_single_node_data,
        run_mpc_perfect,
        solution_diff_heatmaps,
        solution_heatmaps,
        validate_solution_dynamics,
    )


@app.cell(hide_code=True)
def _(mo):
    data_start_input = mo.ui.text(value="2019", label="data start")
    data_end_input = mo.ui.text(value="2019", label="data end")
    add_abnormal_event = mo.ui.switch(label="add abnormal weather event", value = True)
    event_start_input = mo.ui.text(value="2019-08-15", label="event start")
    event_duration_input = mo.ui.number(start=1, stop=90, step=1, label="event duration [days]", value=2)
    event_load_factor_sldr = mo.ui.number(start=0.1, stop=5.0, step=0.05, label="event load factor", value=1.5)
    event_pv_factor_sldr = mo.ui.number(start=0.0, stop=1.0, step=0.05, label="event PV factor", value=0.25)
    mo.vstack(
        [
            add_abnormal_event,
            mo.vstack(
                [
                    data_start_input,
                    data_end_input,
                    event_start_input,
                    event_duration_input,
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
        event_duration_input,
        event_load_factor_sldr,
        event_pv_factor_sldr,
        event_start_input,
    )


@app.cell(hide_code=True)
def _(mo):
    alpha_sldr = mo.ui.number(start=0, stop=50, step=0.25, label="alpha", value=1.25, full_width=True)
    beta_sldr = mo.ui.number(start=0, stop=50, step=0.25, label="beta", value=0.5, full_width=True)
    lambda_sldr = mo.ui.number(start=0, stop=50, step=0.25, label="lambda", value=20.0, full_width=True)
    mu_exp_sldr = mo.ui.number(start=-15, stop=2, step=0.5, label="mu (log base 10)", value=-2, full_width=True)
    G_sldr = mo.ui.number(start=0, step=1, label="G [GW]", value=1, full_width=True)
    Q_sldr = mo.ui.number(start=0, stop=300, step=1, label="battery capacity [GWh]", value=8, full_width=True)
    bat_hours_sldr = mo.ui.number(
        start=0, stop=300, step=1, label="battery number of hours for full discharge", value=4, full_width=True
    )
    round_trip_efficiency_sldr = mo.ui.number(
        start=0.75, stop=1, label="round trip efficiency", value=0.95, full_width=True
    )
    monthly_soc_loss_sldr = mo.ui.number(
        start=0, stop=10, step=0.5, label="soc loss [% per month]", value=2, full_width=True
    )
    form = mo.md(
        """{alpha}\n{beta}\n{lambd}\n{mu_exp}\n{G}\n{Q}\n{bat_hours}\n{round_trip_efficiency}\n{monthly_soc_loss}"""
    ).batch(
        alpha=alpha_sldr,
        beta=beta_sldr,
        lambd=lambda_sldr,
        mu_exp=mu_exp_sldr,
        G=G_sldr,
        Q=Q_sldr,
        bat_hours=bat_hours_sldr,
        round_trip_efficiency=round_trip_efficiency_sldr,
        monthly_soc_loss=monthly_soc_loss_sldr,
    )
    form
    return (form,)


@app.cell(hide_code=True)
def _(
    add_abnormal_event,
    data_end_input,
    data_path,
    data_start_input,
    event_duration_input,
    event_load_factor_sldr,
    event_pv_factor_sldr,
    event_start_input,
    form,
    process_single_node_data,
):
    l, R, shortfall, tidx, daily_df, event_mask, event_shortfall_stats = process_single_node_data(
        form.value["G"],
        data_path=data_path,
        data_start=data_start_input.value,
        data_end=data_end_input.value,
        add_event=add_abnormal_event.value,
        event_start=event_start_input.value,
        event_duration_days=int(event_duration_input.value),
        event_load_factor=event_load_factor_sldr.value,
        event_pv_factor=event_pv_factor_sldr.value,
        verbose=True,
    )
    daily_df.plot(y=["load[MW]", "pv[MW]", "wind[MW]"])
    return R, event_mask, event_shortfall_stats, l, shortfall, tidx


@app.cell(hide_code=True)
def _(form, l, make_one_shot):
    one_shot_problem = make_one_shot(
        alpha=form.value["alpha"],
        beta=form.value["beta"],
        lamb=form.value["lambd"],
        mu=10 ** form.value["mu_exp"],
        T=len(l),
    )
    return (one_shot_problem,)


@app.cell(hide_code=True)
def _(
    R,
    form,
    l,
    load_one_shot_problem_data,
    one_shot_problem,
    validate_solution_dynamics,
):
    load_one_shot_problem_data(
        problem=one_shot_problem,
        l=l,
        R=R,
        G=form.value["G"],
        Q=form.value["Q"],
        B=form.value["Q"] / form.value["bat_hours"],
        q0=form.value["Q"] / 2,
        round_trip_efficiency=form.value["round_trip_efficiency"],
        monthly_soc_loss=form.value["monthly_soc_loss"],
    )
    one_shot_problem.solve(solver="CLARABEL")
    print(f"one-shot status: {one_shot_problem.status}, objective: {one_shot_problem.value:.4f}")
    print(f"dynamics valid: {validate_solution_dynamics(one_shot_problem)}")
    one_shot_solution = {k: v.value for k, v in one_shot_problem.var_dict.items()}
    return (one_shot_solution,)


@app.cell(hide_code=True)
def _(compute_partition, form, np, one_shot_solution, pd, plt, tidx):
    _decouple_points, _is_loadshed = compute_partition(one_shot_solution["q"][1:], tidx, form.value["Q"])
    _all_points = np.concatenate([[tidx[0]], _decouple_points, [tidx[-1]]])
    _durations_days = np.diff([pd.Timestamp(t) for t in _all_points]) / pd.Timedelta("1D")
    _loadshed_days = _durations_days[_is_loadshed]
    _curtail_days = _durations_days[~_is_loadshed]

    for _label, _days in [
        ("load shed zone (F->E)", _loadshed_days),
        ("non-dispatch curtail zone (E->F)", _curtail_days),
    ]:
        print(f"\n{_label}  (n={len(_days)})")
        print(
            f"  mean={np.mean(_days):.2f}d  median={np.median(_days):.2f}d  std={np.std(_days):.2f}d  min={np.min(_days):.2f}d  max={np.max(_days):.2f}d"
        )

    _fig, (_ax1, _ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    _ax1.hist(_loadshed_days, bins=20, color="orange", edgecolor="white", linewidth=0.5, alpha=0.8)
    _ax1.set(ylabel="count", title="load shed zone duration (F->E)")
    _ax2.hist(_curtail_days, bins=20, color="steelblue", edgecolor="white", linewidth=0.5, alpha=0.8)
    _ax2.set(ylabel="count", title="non-dispatch curtail zone duration (E->F)", xlabel="segment length [days]")
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    H_sldr = mo.ui.number(start=1, step=1, label="MPC horizon H (time steps)", value=72, full_width=True)
    q_target_sldr = mo.ui.number(
        start=0.0, stop=1.0, step=0.05, label="q_target (fraction of Q)", value=0.5, full_width=True
    )
    gamma_exp_sldr = mo.ui.number(start=-5, stop=0, step=1, label="gamma (log base 10)", value=-2, full_width=True)
    form_mpc = mo.md("""{H}\n{q_target}\n{gamma_exp}""").batch(
        H=H_sldr,
        q_target=q_target_sldr,
        gamma_exp=gamma_exp_sldr,
    )
    form_mpc
    return (form_mpc,)


@app.cell(hide_code=True)
def _(R, form, form_mpc, l, run_mpc_perfect):
    mpc_solution = run_mpc_perfect(
        l=l,
        R=R,
        G=form.value["G"],
        Q=form.value["Q"],
        B=form.value["Q"] / form.value["bat_hours"],
        alpha=form.value["alpha"],
        beta=form.value["beta"],
        lamb=form.value["lambd"],
        gamma=10 ** form_mpc.value["gamma_exp"],
        mu=10 ** form.value["mu_exp"],
        round_trip_efficiency=form.value["round_trip_efficiency"],
        monthly_soc_loss=form.value["monthly_soc_loss"],
        q_init=form.value["Q"] / 2,
        q_target=form_mpc.value["q_target"],
        H=form_mpc.value["H"],
    )
    return (mpc_solution,)


@app.cell(hide_code=True)
def _(
    event_mask,
    form,
    get_metrics_of_interest,
    mpc_solution,
    np,
    one_shot_solution,
):
    _kwargs = dict(
        lamb=form.value["lambd"],
        alpha=form.value["alpha"],
        beta=form.value["beta"],
        mu=10 ** form.value["mu_exp"],
        stress_mask=event_mask,
    )
    _os_m = get_metrics_of_interest(
        s=one_shot_solution["s"],
        g=one_shot_solution["g"],
        b=one_shot_solution["b"],
        c=one_shot_solution["c"],
        **_kwargs,
    )
    _mpc_m = get_metrics_of_interest(
        s=mpc_solution["s"], g=mpc_solution["g"], b=mpc_solution["b"], c=mpc_solution["c"], **_kwargs
    )
    for _key in _os_m:
        _os, _mpc = _os_m[_key], _mpc_m[_key]
        _rel = (_mpc - _os) / _os if np.round(_os, decimals=3) != 0 else float("nan")
        print(f"{_key}: one-shot={_os:.3f}  mpc perfect={_mpc:.3f}  delta={_mpc - _os:+.3f}  ({_rel:+.1%})")
    return


@app.cell(hide_code=True)
def _(
    add_abnormal_event,
    event_duration_input,
    event_start_input,
    mo,
    pd,
    tidx,
):
    _steps_per_day = int(pd.Timedelta("1D") / (tidx[1] - tidx[0]))
    if add_abnormal_event.value:
        _event_start_idx = int(tidx.searchsorted(pd.Timestamp(event_start_input.value)))
        _duration = int(event_duration_input.value)
        _default_start_idx = max(0, _event_start_idx - _duration * _steps_per_day // 2)
        _default_start = str(pd.Timestamp(tidx[_default_start_idx]).date())
        _default_days = 2 * _duration
    else:
        _default_start = str(pd.Timestamp(tidx[0]).date())
        _default_days = 7
    plot_start_date = mo.ui.date(value=_default_start, label="plot start")
    plot_length_days = mo.ui.number(start=1, step=1, label="plot length [days]", value=_default_days)
    mo.output.append(mo.vstack([plot_start_date, plot_length_days]))
    return plot_length_days, plot_start_date


@app.cell(hide_code=True)
def _(np, pd, plot_length_days, plot_start_date, tidx):
    _steps_per_day = int(pd.Timedelta("1D") / (tidx[1] - tidx[0]))
    _start_idx = int(tidx.searchsorted(pd.Timestamp(str(plot_start_date.value))))
    s = np.s_[_start_idx : _start_idx + int(plot_length_days.value) * _steps_per_day]
    return (s,)


@app.cell(hide_code=True)
def _(
    R,
    add_abnormal_event,
    event_duration_input,
    event_start_input,
    form,
    l,
    pd,
    plot_timeseries,
    s,
    shortfall,
    tidx,
):
    _event_end = str((pd.Timestamp(event_start_input.value) + pd.Timedelta(days=int(event_duration_input.value) - 1)).date())
    _event_dates = [event_start_input.value, _event_end] if add_abnormal_event.value else None
    plot_timeseries(l=l, R=R, shortfall=shortfall, tidx=tidx, s=s, G=form.value["G"], event_dates=_event_dates)
    return


@app.cell(hide_code=True)
def _(
    add_abnormal_event,
    event_duration_input,
    event_start_input,
    form,
    one_shot_solution,
    pd,
    plot_solution,
    s,
    tidx,
):
    _event_end = str((pd.Timestamp(event_start_input.value) + pd.Timedelta(days=int(event_duration_input.value) - 1)).date())
    _event_dates = [event_start_input.value, _event_end] if add_abnormal_event.value else None
    plot_solution(
        solution=one_shot_solution,
        tidx=tidx,
        s=s,
        Q=form.value["Q"],
        B=form.value["Q"] / form.value["bat_hours"],
        alpha=form.value["alpha"],
        beta=form.value["beta"],
        efficiency=form.value["round_trip_efficiency"],
        supertitle="one-shot solution",
        event_dates=_event_dates,
    )
    return


@app.cell(hide_code=True)
def _(
    add_abnormal_event,
    event_duration_input,
    event_start_input,
    form,
    form_mpc,
    mpc_solution,
    pd,
    plot_solution,
    s,
    tidx,
):
    _event_end = str((pd.Timestamp(event_start_input.value) + pd.Timedelta(days=int(event_duration_input.value) - 1)).date())
    _event_dates = [event_start_input.value, _event_end] if add_abnormal_event.value else None
    plot_solution(
        solution=mpc_solution,
        tidx=tidx,
        s=s,
        Q=form.value["Q"],
        B=form.value["Q"] / form.value["bat_hours"],
        alpha=form.value["alpha"],
        beta=form.value["beta"],
        efficiency=form.value["round_trip_efficiency"],
        supertitle=f"MPC perfect solution (H={form_mpc.value['H']})",
        event_dates=_event_dates,
    )
    return


@app.cell(hide_code=True)
def _(
    add_abnormal_event,
    event_duration_input,
    event_start_input,
    form,
    mo,
    one_shot_solution,
    pd,
    solution_heatmaps,
    tidx,
):
    _event_end = str((pd.Timestamp(event_start_input.value) + pd.Timedelta(days=int(event_duration_input.value) - 1)).date())
    _event_dates = [event_start_input.value, _event_end] if add_abnormal_event.value else None
    mo.output.append(
        solution_heatmaps(
            one_shot_solution,
            tidx,
            Q=form.value["Q"],
            B=form.value["Q"] / form.value["bat_hours"],
            G=form.value["G"],
            label="one-shot",
            event_dates=_event_dates,
        )
    )
    return


@app.cell(hide_code=True)
def _(
    add_abnormal_event,
    event_duration_input,
    event_start_input,
    form,
    mo,
    mpc_solution,
    pd,
    solution_heatmaps,
    tidx,
):
    _event_end = str((pd.Timestamp(event_start_input.value) + pd.Timedelta(days=int(event_duration_input.value) - 1)).date())
    _event_dates = [event_start_input.value, _event_end] if add_abnormal_event.value else None
    mo.output.append(
        solution_heatmaps(
            mpc_solution,
            tidx,
            Q=form.value["Q"],
            B=form.value["Q"] / form.value["bat_hours"],
            G=form.value["G"],
            label="MPC perfect",
            event_dates=_event_dates,
        )
    )
    return


@app.cell(hide_code=True)
def _(
    add_abnormal_event,
    event_duration_input,
    event_start_input,
    form,
    mo,
    mpc_solution,
    one_shot_solution,
    pd,
    solution_diff_heatmaps,
    tidx,
):
    _event_end = str((pd.Timestamp(event_start_input.value) + pd.Timedelta(days=int(event_duration_input.value) - 1)).date())
    _event_dates = [event_start_input.value, _event_end] if add_abnormal_event.value else None
    mo.output.append(
        solution_diff_heatmaps(
            mpc_solution,
            one_shot_solution,
            tidx,
            Q=form.value["Q"],
            B=form.value["Q"] / form.value["bat_hours"],
            G=form.value["G"],
            label_a="MPC perfect",
            label_b="one-shot",
            event_dates=_event_dates,
        )
    )
    return


if __name__ == "__main__":
    app.run()
