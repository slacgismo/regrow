import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import pathlib
    import sys
    import json

    import marimo as mo
    import numpy as np
    import pandas as pd

    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
    from controllers.data_utils import process_single_node_data
    from controllers.mpc_perfect import run_mpc_perfect
    from experiments.battery_sizing.battery_sizing import tune_mpc_params

    data_path = str(pathlib.Path(__file__).parent.parent.parent.parent / "single_node_data.csv")
    return (
        data_path,
        json,
        mo,
        np,
        pathlib,
        pd,
        process_single_node_data,
        tune_mpc_params,
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
    l, R, shortfall, tidx, daily_df, event_mask = process_single_node_data(
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
    return G, R, event_mask, l


@app.cell(hide_code=True)
def _(mo):
    alpha_sldr = mo.ui.slider(start=0, stop=50, step=0.25, label="alpha", value=1.25, full_width=True)
    beta_sldr = mo.ui.slider(start=0, stop=50, step=0.25, label="beta", value=0.5, full_width=True)
    lambda_sldr = mo.ui.slider(start=0, stop=50, step=0.25, label="lambda", value=20.0, full_width=True)
    mu_exp_sldr = mo.ui.slider(start=-6, stop=2, step=0.5, label="mu (log base 10)", value=-2, full_width=True)
    Q_sldr = mo.ui.number(start=0, stop=300, step=1, label="battery capacity [GWh]", value=4, full_width=True)
    bat_hours_sldr = mo.ui.number(
        start=0, stop=300, step=1, label="battery number of hours for full discharge", value=4, full_width=True
    )
    H_sldr = mo.ui.number(start=1, stop=8760, step=1, label="MPC horizon H (time steps)", value=24, full_width=True)
    form = mo.md("""{alpha}\n{beta}\n{lamb}\n{mu_exp}\n{Q}\n{bat_hours}\n{H}""").batch(
        alpha=alpha_sldr,
        beta=beta_sldr,
        lamb=lambda_sldr,
        mu_exp=mu_exp_sldr,
        Q=Q_sldr,
        bat_hours=bat_hours_sldr,
        H=H_sldr,
    )
    form
    return (form,)


@app.cell
def _(np):
    gamma_list = np.logspace(start=-4, stop=-1, num=4)
    q_target_list = np.array([0.25, 0.5, 0.75, 1])
    return gamma_list, q_target_list


@app.cell
def _(G, R, event_mask, form, gamma_list, l, q_target_list, tune_mpc_params):
    best_row = tune_mpc_params(
        l=l,
        R=R,
        G=G,
        Q=form.value['Q'],
        B=form.value['Q'] / form.value['bat_hours'],
        alpha=form.value['alpha'],
        beta=form.value['beta'],
        lamb=form.value['lamb'],
        mu=10**form.value['mu_exp'],
        H=form.value['H'],
        gamma_list=gamma_list,
        q_target_list=q_target_list,
        stress_mask=event_mask,
        plot_list=['objective', 'normal objective', 'normal total load shedding', 'stress total load shedding'])
    return


@app.cell
def _(np):
    Q_list = 2**np.arange(0,6)
    H_list = 24*np.array([1,2,3])
    return H_list, Q_list


@app.cell
def _(
    G,
    H_list,
    Q_list,
    R,
    add_abnormal_event,
    data_end_input,
    data_start_input,
    event_end_input,
    event_load_factor_sldr,
    event_mask,
    event_pv_factor_sldr,
    event_start_input,
    form,
    gamma_list,
    json,
    l,
    np,
    pathlib,
    pd,
    q_target_list,
    tune_mpc_params,
):
    sizing_config = {
        "data": {
            "data_start": data_start_input.value,
            "data_end": data_end_input.value,
            "add_event": add_abnormal_event.value,
            "event_start": event_start_input.value,
            "event_end": event_end_input.value,
            "event_load_factor": event_load_factor_sldr.value,
            "event_pv_factor": event_pv_factor_sldr.value,
        },
        "fixed_params": {
            "G": G,
            "bat_hours": form.value['bat_hours'],
            "alpha": form.value['alpha'],
            "beta": form.value['beta'],
            "lamb": form.value['lamb'],
            "mu_exp": form.value['mu_exp'],
            "mu": 10**form.value['mu_exp'],
        },
        "sweep": {
            "Q_list": np.array(Q_list).tolist(),
            "H_list": np.array(H_list).tolist(),
            "gamma_list": gamma_list.tolist(),
            "q_target_list": q_target_list.tolist(),
        },
    }
    pathlib.Path(__file__).parent.joinpath("sizing_sweep_config.json").write_text(json.dumps(sizing_config, indent=2))
    rows = pd.DataFrame([
        tune_mpc_params(
            l=l,
            R=R,
            G=G,
            Q=Q,
            B=Q / form.value['bat_hours'],
            alpha=form.value['alpha'],
            beta=form.value['beta'],
            lamb=form.value['lamb'],
            mu=10**form.value['mu_exp'],
            H=H,
            gamma_list=gamma_list,
            q_target_list=q_target_list,
            stress_mask=event_mask,
        ) for Q in Q_list for H in H_list
    ])
    rows.to_csv(pathlib.Path(__file__).parent / "sizing_sweep_results.csv", index=False)
    return (rows,)


@app.cell
def _(rows):
    rows
    return


if __name__ == "__main__":
    app.run()
