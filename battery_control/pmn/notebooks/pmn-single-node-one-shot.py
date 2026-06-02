import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import pathlib
    import sys

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    from controllers.one_shot import make_one_shot, load_one_shot_problem_data
    from controllers.constraints import validate_solution_dynamics
    from controllers.data_utils import process_single_node_data
    from plot_utils import compute_partition, plot_solution, solution_heatmaps

    data_path = str(pathlib.Path(__file__).parent.parent.parent / "single_node_data.csv")
    return (
        compute_partition,
        data_path,
        load_one_shot_problem_data,
        make_one_shot,
        mo,
        np,
        pd,
        plot_solution,
        plt,
        process_single_node_data,
        solution_heatmaps,
        validate_solution_dynamics,
    )


@app.cell
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
def _(mo):
    alpha_sldr = mo.ui.number(start=0, stop=50, step=0.25, label="alpha", value=1.25, full_width=True)
    beta_sldr = mo.ui.number(start=0, stop=50, step=0.25, label="beta", value=0.5, full_width=True)
    lambda_sldr = mo.ui.number(start=0, stop=50, step=0.25, label="lambda", value=20.0, full_width=True)
    mu_exp_sldr = mo.ui.number(start=-15, stop=2, step=0.5, label="mu (log base 10)", value=-3, full_width=True)
    G_sldr = mo.ui.number(start=0, step=1, label="G [GW]", value=1, full_width=True)
    Q_sldr = mo.ui.number(start=0, stop=300, step=1, label="battery capacity [GWh]", value=4, full_width=True)
    bat_hours_sldr = mo.ui.number(
        start=0, stop=300, step=1, label="battery number of hours for full discharge", value=3, full_width=True
    )
    round_trip_efficiency_sldr = mo.ui.number(start=0.7, stop=1, label="round trip efficiency", value=0.95, full_width=True)
    monthly_soc_loss_sldr = mo.ui.number(start=0, stop=10, step=0.5, label="soc loss [% per month]", value=1, full_width=True)
    form = mo.md("""{alpha}\n{beta}\n{lambd}\n{mu_exp}\n{G}\n{Q}\n{bat_hours}\n{round_trip_efficiency}\n{monthly_soc_loss}""").batch(
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
    form,
    process_single_node_data,
):
    l, R, shortfall, tidx, daily_df, event_mask = process_single_node_data(
        form.value["G"],
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
    return R, l, tidx


@app.cell
def _(form, l, make_one_shot):
    one_shot_problem = make_one_shot(
        alpha=form.value["alpha"],
        beta=form.value["beta"],
        lamb=form.value["lambd"],
        mu=10 ** form.value["mu_exp"],
        T=len(l),
    )
    return (one_shot_problem,)


@app.cell
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
    print(f"status: {one_shot_problem.status}, objective: {one_shot_problem.value:.4f}")
    print(f"dynamics valid: {validate_solution_dynamics(one_shot_problem)}")
    solution = {k: v.value for k, v in one_shot_problem.var_dict.items()}
    return (solution,)


@app.cell
def _(add_abnormal_event, event_end_input, event_start_input, mo, pd, tidx):
    _steps_per_day = int(pd.Timedelta("1D") / (tidx[1] - tidx[0]))
    if add_abnormal_event.value:
        _event_start_idx = int(tidx.searchsorted(pd.Timestamp(event_start_input.value)))
        _event_end_idx = int(tidx.searchsorted(pd.Timestamp(event_end_input.value)))
        _event_duration_days = max(1, (_event_end_idx - _event_start_idx) // _steps_per_day)
        _default_start_idx = max(0, _event_start_idx - _event_duration_days * _steps_per_day // 2)
        _default_start = str(pd.Timestamp(tidx[_default_start_idx]).date())
        _default_days = 2 * _event_duration_days
    else:
        _default_start = str(pd.Timestamp(tidx[0]).date())
        _default_days = 7
    plot_start_date = mo.ui.date(value=_default_start, label="plot start")
    plot_length_days = mo.ui.number(start=1, step=1, label="plot length [days]", value=_default_days)
    mo.output.append(mo.vstack([plot_start_date, plot_length_days]))
    return plot_length_days, plot_start_date


@app.cell
def _(
    form,
    np,
    pd,
    plot_length_days,
    plot_solution,
    plot_start_date,
    solution,
    tidx,
):
    _steps_per_day = int(pd.Timedelta("1D") / (tidx[1] - tidx[0]))
    _start_idx = int(tidx.searchsorted(pd.Timestamp(str(plot_start_date.value))))
    _s = np.s_[_start_idx : _start_idx + int(plot_length_days.value) * _steps_per_day]
    _fig = plot_solution(

        solution=solution,
        tidx=tidx,
        s=_s,
        Q=form.value["Q"],
        B=form.value["Q"] / form.value["bat_hours"],
        alpha=form.value["alpha"],
        beta=form.value["beta"],
        efficiency=form.value["round_trip_efficiency"],
    )
    _fig
    return


@app.cell
def _(compute_partition, form, np, pd, plt, solution, tidx):
    _decouple_points, _is_loadshed = compute_partition(solution["q"][1:], tidx, form.value["Q"])
    _all_points = np.concatenate([[tidx[0]], _decouple_points, [tidx[-1]]])
    _durations_days = np.diff([pd.Timestamp(t) for t in _all_points]) / pd.Timedelta("1D")
    _loadshed_days = _durations_days[_is_loadshed]
    _curtail_days = _durations_days[~_is_loadshed]

    for _label, _days in [("load shed zone (F to E)", _loadshed_days), ("non-dispatch curtail zone (E toF)", _curtail_days)]:
        print(f"\n{_label}  (n={len(_days)})")
        print(f"  mean={np.mean(_days):.2f}d  median={np.median(_days):.2f}d  std={np.std(_days):.2f}d  min={np.min(_days):.2f}d  max={np.max(_days):.2f}d")

    _fig, (_ax1, _ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    _ax1.hist(_loadshed_days, bins=20, color="orange", edgecolor="white", linewidth=0.5, alpha=0.8)
    _ax1.set(ylabel="count", title="load shed zone (F to E)")
    _ax2.hist(_curtail_days, bins=20, color="steelblue", edgecolor="white", linewidth=0.5, alpha=0.8)
    _ax2.set(ylabel="count", title="non-dispatch curtail zone (E to F)", xlabel="segment length [days]")
    plt.tight_layout()
    _fig
    return


@app.cell
def _(form, mo, solution, solution_heatmaps, tidx):
    mo.output.append(solution_heatmaps(solution, tidx, Q=form.value["Q"], B=form.value["Q"] / form.value["bat_hours"], G=form.value["G"]))
    return


if __name__ == "__main__":
    app.run()
