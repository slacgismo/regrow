import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from controllers.data_utils import process_single_node_data
from controllers.metrics import get_metrics_of_interest
from controllers.mpc_perfect import run_mpc_perfect
from controllers.one_shot import load_one_shot_problem_data, make_one_shot
from tqdm import tqdm


def _run_pair(fixed_kwargs, stress_mask, pair):
    gamma, q_target = pair
    mpc_solution = run_mpc_perfect(gamma=gamma, q_target=q_target, **fixed_kwargs)
    mpc_metrics = get_metrics_of_interest(
        s=mpc_solution["s"],
        g=mpc_solution["g"],
        b=mpc_solution["b"],
        c=mpc_solution["c"],
        lamb=fixed_kwargs["lamb"],
        alpha=fixed_kwargs["alpha"],
        beta=fixed_kwargs["beta"],
        mu=fixed_kwargs["mu"],
        delta=fixed_kwargs["delta"],
        stress_mask=stress_mask,
    )
    return {"gamma": gamma, "q_target": q_target, **mpc_metrics}


def tune_mpc_params(
    l,
    R,
    G,
    Q,
    B,
    alpha,
    beta,
    lamb,
    mu,
    H,
    gamma_list,
    q_target_list,
    round_trip_efficiency,
    monthly_soc_loss,
    q_init=None,
    delta=1,
    solver="CLARABEL",
    stress_mask=None,
    plot_list=None,
    max_workers=4,
):
    if q_init is None:
        q_init = Q / 2

    fixed_kwargs = dict(
        l=l,
        R=R,
        G=G,
        Q=Q,
        B=B,
        alpha=alpha,
        beta=beta,
        lamb=lamb,
        mu=mu,
        q_init=q_init,
        H=H,
        round_trip_efficiency=round_trip_efficiency,
        monthly_soc_loss=monthly_soc_loss,
        delta=delta,
        solver=solver,
        disable_progress_bar=True,
    )

    pairs = list(product(gamma_list, q_target_list))
    run = partial(_run_pair, fixed_kwargs, stress_mask)

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        rows = list(tqdm(pool.map(run, pairs), total=len(pairs)))

    df = pd.DataFrame(rows)
    if plot_list is not None:
        for metric in plot_list:
            if metric not in df.columns:
                raise ValueError(f"Invalid metric {metric} in plot list")
            pivot = df.pivot(index="gamma", columns="q_target", values=metric)
            sns.heatmap(pivot, annot=True, fmt=".3g", cmap="Blues")
            plt.title(metric)
            plt.show()
    best_idx = df["objective"].idxmin()
    best_row = df.loc[best_idx].to_dict()
    df.insert(0, "H", H)
    df.insert(0, "Q", Q)
    return {"Q": Q, "H": H, **best_row}, df


def run_sizing_experiment(experiment_name):
    # data
    G = 1
    DATA_START = "2019"
    DATA_END = "2019"
    ADD_EVENT = True
    EVENT_START = "2019-08-15"
    EVENT_DURATION_DAYS = 2
    EVENT_LOAD_FACTOR = 1.5
    EVENT_PV_FACTOR = 0.25

    # fixed battery and objective parameters
    BAT_HOURS = 4
    ROUND_TRIP_EFFICIENCY = 0.95
    MONTHLY_SOC_LOSS = 2
    ALPHA = 1.25
    BETA = 0.5
    LAMB = 20.0
    MU = 1e-2

    # point sweeps
    Q_LIST = 2 ** np.arange(0, 6)
    H_LIST = 24 * np.array([1, 2, 3, 4])

    # per point sweeps
    GAMMA_LIST = np.logspace(-4, -1, num=4)
    Q_TARGET_LIST = np.array([0.5, 0.75, 1.0])

    config = {
        "data": {
            "data_start": DATA_START,
            "data_end": DATA_END,
            "add_event": ADD_EVENT,
            "event_start": EVENT_START,
            "event_duration_days": EVENT_DURATION_DAYS,
            "event_load_factor": EVENT_LOAD_FACTOR,
            "event_pv_factor": EVENT_PV_FACTOR,
        },
        "fixed_params": {
            "G": G,
            "bat_hours": BAT_HOURS,
            "efficiency": ROUND_TRIP_EFFICIENCY,
            "soc_loss": MONTHLY_SOC_LOSS,
            "alpha": ALPHA,
            "beta": BETA,
            "lamb": LAMB,
            "mu": MU,
            "delta": 1,
        },
        "sweep": {
            "Q_list": Q_LIST.tolist(),
            "H_list": H_LIST.tolist(),
            "gamma_list": GAMMA_LIST.tolist(),
            "q_target_list": Q_TARGET_LIST.tolist(),
        },
    }

    data_path = str(pathlib.Path(__file__).parent.parent.parent.parent / "single_node_data.csv")
    l, R, _, _, _, event_mask, event_shortfall_stats = process_single_node_data(
        G,
        data_path=data_path,
        data_start=DATA_START,
        data_end=DATA_END,
        add_event=ADD_EVENT,
        event_start=EVENT_START,
        event_duration_days=EVENT_DURATION_DAYS,
        event_load_factor=EVENT_LOAD_FACTOR,
        event_pv_factor=EVENT_PV_FACTOR,
    )

    out_dir = pathlib.Path(__file__).parent / experiment_name
    out_dir.mkdir(exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    if event_shortfall_stats is not None:
        _stats = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in event_shortfall_stats.items()}
        (out_dir / "event_shortfall_stats.json").write_text(json.dumps(_stats, indent=2))
    results_path = out_dir / "results.csv"
    tune_results_path = out_dir / "tune_results.csv"
    one_shot_path = out_dir / "one_shot_results.csv"
    one_shot_problem = make_one_shot(alpha=ALPHA, beta=BETA, lamb=LAMB, mu=MU, T=len(l))
    total = len(Q_LIST) * len(H_LIST)
    rows = []
    tune_rows = []
    one_shot_rows = []
    for Q in Q_LIST:
        print(f"[Q={Q}] running one-shot...", flush=True)
        load_one_shot_problem_data(
            one_shot_problem,
            l=l,
            R=R,
            G=G,
            Q=Q,
            B=Q / BAT_HOURS,
            q0=Q / 2,
            round_trip_efficiency=ROUND_TRIP_EFFICIENCY,
            monthly_soc_loss=MONTHLY_SOC_LOSS,
        )
        one_shot_problem.solve(solver="CLARABEL")
        sol = one_shot_problem.var_dict
        one_shot_rows.append(
            {
                "Q": Q,
                **get_metrics_of_interest(
                    s=sol["s"].value,
                    g=sol["g"].value,
                    b=sol["b"].value,
                    c=sol["c"].value,
                    lamb=LAMB,
                    alpha=ALPHA,
                    beta=BETA,
                    mu=MU,
                    stress_mask=event_mask,
                ),
            }
        )
        pd.DataFrame(one_shot_rows).to_csv(one_shot_path, index=False)
        for H in H_LIST:
            i = len(rows) + 1
            print(f"[{i}/{total}] Q={Q}, H={H}", flush=True)
            best_row, tune_df = tune_mpc_params(
                l=l,
                R=R,
                G=G,
                Q=Q,
                B=Q / BAT_HOURS,
                alpha=ALPHA,
                beta=BETA,
                lamb=LAMB,
                mu=MU,
                H=H,
                round_trip_efficiency=ROUND_TRIP_EFFICIENCY,
                monthly_soc_loss=MONTHLY_SOC_LOSS,
                gamma_list=GAMMA_LIST,
                q_target_list=Q_TARGET_LIST,
                stress_mask=event_mask,
            )
            rows.append(best_row)
            tune_rows.append(tune_df)
            pd.DataFrame(rows).to_csv(results_path, index=False)
            pd.concat(tune_rows, ignore_index=True).to_csv(tune_results_path, index=False)

    _make_plots(pd.read_csv(results_path), pd.read_csv(one_shot_path), out_dir)


def _plot_metrics(metrics, mpc_df, one_shot_df, axes):
    H_values = sorted(mpc_df["H"].unique())
    for ax, metric in zip(axes, metrics):
        for H in H_values:
            sub = mpc_df[mpc_df["H"] == H].sort_values("Q")
            ax.plot(sub["Q"], sub[metric], marker="o", label=f"MPC H={H}")
        os_ = one_shot_df.sort_values("Q")
        if metric in one_shot_df.columns:
            ax.plot(os_["Q"], os_[metric], marker="s", linestyle="--", color="black", label="one-shot")
        ax.set_xlabel("Q [GWh]")
        ax.set_title(metric)
        ax.set_xscale("log", base=2)
        ax.legend(fontsize=7)
    for ax in axes[len(metrics):]:
        ax.set_visible(False)


def _make_plots(mpc_df, one_shot_df, out_dir):
    H_values = sorted(mpc_df["H"].unique())

    fig_obj, ax_obj = plt.subplots(figsize=(6, 4))
    for H in H_values:
        sub = mpc_df[mpc_df["H"] == H].sort_values("Q")
        ax_obj.plot(sub["Q"], sub["objective"], marker="o", label=f"MPC H={H}")
    os_ = one_shot_df.sort_values("Q")
    ax_obj.plot(os_["Q"], os_["objective"], marker="s", linestyle="--", color="black", label="one-shot")
    ax_obj.set_xlabel("Q [GWh]")
    ax_obj.set_title("objective")
    ax_obj.set_xscale("log", base=2)
    ax_obj.legend(fontsize=7)
    plt.tight_layout()
    fig_obj.savefig(out_dir / "battery-sizing-objective.png", bbox_inches="tight", dpi=600)
    plt.close(fig_obj)

    normal_metrics = [c for c in mpc_df.columns if c.startswith("normal ")]
    stress_metrics = [c for c in mpc_df.columns if c.startswith("stress ")]
    n_cols = 5
    fig_breakdown, axes = plt.subplots(2, n_cols, figsize=(25, 8))
    _plot_metrics(normal_metrics, mpc_df, one_shot_df, axes[0])
    _plot_metrics(stress_metrics, mpc_df, one_shot_df, axes[1])
    plt.tight_layout()
    fig_breakdown.savefig(out_dir / "battery-sizing-breakdown.png", bbox_inches="tight", dpi=600)
    plt.close(fig_breakdown)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name")
    args = parser.parse_args()
    run_sizing_experiment(args.experiment_name)
