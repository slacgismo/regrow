import argparse
import json
import pathlib
import sys
from datetime import date, timedelta

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from controllers.constraints import validate_solution_dynamics
from controllers.data_utils import process_single_node_data
from controllers.one_shot import load_one_shot_problem_data, make_one_shot

G = 1
BAT_HOURS = 4
EFFICIENCY = 0.98
SOC_LOSS = 1e-5
ALPHA = 1.25
BETA = 0.5
LAMB = 20.0
Q = 10  # base

Q_LIST = [4, 8, 16, 32]  # Q sweep

MU_LIST = np.logspace(-6, 0, num=7)

_EVENT_START = date(2019, 8, 15)
_EVENT_LOAD_FACTOR = 1.5
_EVENT_PV_FACTOR = 0.25
_EVENT_DURATIONS_DAYS = [1, 2, 4, 8]


CONFIGS = [
    {"name": "3yr", "data_start": "2018", "data_end": "2020", "add_event": False},
    {"name": "2018", "data_start": "2018", "data_end": "2018", "add_event": False},
    {"name": "2019", "data_start": "2019", "data_end": "2019", "add_event": False},
    {"name": "2020", "data_start": "2020", "data_end": "2020", "add_event": False},
] + [
    {
        "name": f"event_{d}d",
        "data_start": "2019",
        "data_end": "2019",
        "add_event": True,
        "event_start": str(_EVENT_START),
        "event_end": str(_EVENT_START + timedelta(days=d)),
        "event_load_factor": _EVENT_LOAD_FACTOR,
        "event_pv_factor": _EVENT_PV_FACTOR,
    }
    for d in _EVENT_DURATIONS_DAYS
]


def _sweep_mu(l, R, n_days, q=Q):
    B = q / BAT_HOURS
    rows = []
    for mu in MU_LIST:
        problem = make_one_shot(
            alpha=ALPHA, beta=BETA, lamb=LAMB, mu=mu, T=len(l),
            efficiency=EFFICIENCY, soc_loss=SOC_LOSS,
        )
        load_one_shot_problem_data(problem, l=l, R=R, G=G, Q=q, B=B, q0=q / 2)
        problem.solve(solver="CLARABEL")
        if problem.status not in ("optimal", "optimal_inaccurate"):
            print(f"  skipping mu={mu:.2e}: status={problem.status}")
            continue
        if not validate_solution_dynamics(
            problem, Q=q, B=q / BAT_HOURS, efficiency=EFFICIENCY, soc_loss=SOC_LOSS, delta=1
        ):
            print(f"  skipping mu={mu:.2e}: dynamics validation failed")
            continue
        sol = {k: v.value for k, v in problem.var_dict.items()}
        rows.append({
            "mu": mu,
            "Q": q,
            "throughput_per_day": float(np.sum(np.abs(sol["b"])) / n_days),
            "non_battery_cost_per_day": float(
                np.sum(LAMB * sol["s"] + ALPHA * sol["g"] + BETA * sol["g"] ** 2) / n_days
            ),
        })
    return rows


def _plot_group(ax, df, names, labels=None, col="config"):
    for name, label in zip(names, labels or names):
        sub = df[df[col] == name].sort_values("mu")
        ax.plot(sub["throughput_per_day"], sub["non_battery_cost_per_day"], marker="o", label=label)
    ax.set_xlabel("battery throughput [GWh/day]")
    ax.set_ylabel("operational cost (load shed + generation) [/day]")
    ax.legend()


def _plot_group_vs_mu(ax, df, names, labels=None, col="config"):
    for name, label in zip(names, labels or names):
        sub = df[df[col] == name].sort_values("mu")
        ax.plot(sub["mu"], sub["non_battery_cost_per_day"], marker="o", label=label)
    ax.set_xscale("log")
    ax.set_xlabel("mu")
    ax.set_ylabel("operational cost (load shed + generation) [/day]")
    ax.legend()


def _save(fig, path):
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _make_plots(df, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    _plot_group(ax, df, ["3yr", "2018", "2019", "2020"], ["2018-2020", "2018", "2019", "2020"])
    ax.set_title("mu pareto: individual years")
    _save(fig, out_dir / "pareto_years.png")

    event_names = [f"event_{d}d" for d in _EVENT_DURATIONS_DAYS]
    event_labels = [f"{d}d" for d in _EVENT_DURATIONS_DAYS]
    fig, ax = plt.subplots(figsize=(7, 5))
    _plot_group(ax, df, event_names, event_labels)
    ax.set_title("mu pareto: 2019 event duration sweep")
    ax.legend(title="event duration")
    _save(fig, out_dir / "pareto_events.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    _plot_group_vs_mu(ax, df, ["3yr", "2018", "2019", "2020"], ["2018-2020", "2018", "2019", "2020"])
    ax.set_title("mu vs cost: individual years")
    _save(fig, out_dir / "mu_cost_years.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    _plot_group_vs_mu(ax, df, event_names, event_labels)
    ax.set_title("mu vs cost: 2019 event duration sweep")
    ax.legend(title="event duration")
    _save(fig, out_dir / "mu_cost_events.png")

    q_df = df[df["config"] == "3yr_Qsweep"]
    q_vals = sorted(q_df["Q"].unique())
    fig, ax = plt.subplots(figsize=(7, 5))
    _plot_group(ax, q_df, q_vals, labels=[f"Q={q}" for q in q_vals], col="Q")
    ax.set_title("mu pareto: battery size sweep (3yr)")
    ax.legend(title="Q [GWh]")
    _save(fig, out_dir / "pareto_Q.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    _plot_group_vs_mu(ax, q_df, q_vals, labels=[f"Q={q}" for q in q_vals], col="Q")
    ax.set_title("mu vs cost: battery size sweep (3yr)")
    ax.legend(title="Q [GWh]")
    _save(fig, out_dir / "mu_cost_Q.png")


def run(experiment_name):
    data_path = str(pathlib.Path(__file__).parent.parent.parent.parent / "single_node_data.csv")
    out_dir = pathlib.Path(__file__).parent / experiment_name
    out_dir.mkdir(exist_ok=True)

    config = {
        "fixed_params": {
            "G": G,
            "Q": Q,
            "BAT_HOURS": BAT_HOURS,
            "EFFICIENCY": EFFICIENCY,
            "SOC_LOSS": SOC_LOSS,
            "ALPHA": ALPHA,
            "BETA": BETA,
            "LAMB": LAMB,
        },
        "mu_list": MU_LIST.tolist(),
        "Q_list": Q_LIST,
        "configs": CONFIGS,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    all_rows = []
    for i, cfg in enumerate(CONFIGS):
        print(f"[{i+1}/{len(CONFIGS)}] config={cfg['name']}", flush=True)
        l, R, _, tidx, *_ = process_single_node_data(
            G,
            data_path=data_path,
            data_start=cfg["data_start"],
            data_end=cfg["data_end"],
            add_event=cfg.get("add_event", False),
            event_start=cfg.get("event_start", str(_EVENT_START)),
            event_end=cfg.get("event_end", str(_EVENT_START + timedelta(days=5))),
            event_load_factor=cfg.get("event_load_factor", _EVENT_LOAD_FACTOR),
            event_pv_factor=cfg.get("event_pv_factor", _EVENT_PV_FACTOR),
        )
        n_days = len(tidx) * (tidx[1] - tidx[0]) / pd.Timedelta("1D")
        for row in _sweep_mu(l, R, n_days):
            all_rows.append({"config": cfg["name"], **row})

    l_3yr, R_3yr, _, tidx_3yr, *_ = process_single_node_data(
        G, data_path=data_path, data_start="2018", data_end="2020", add_event=False,
        event_start=str(_EVENT_START), event_end=str(_EVENT_START + timedelta(days=5)),
        event_load_factor=_EVENT_LOAD_FACTOR, event_pv_factor=_EVENT_PV_FACTOR,
    )
    n_days_3yr = len(tidx_3yr) * (tidx_3yr[1] - tidx_3yr[0]) / pd.Timedelta("1D")
    for j, q_val in enumerate(Q_LIST):
        print(f"[Q sweep {j+1}/{len(Q_LIST)}] Q={q_val}", flush=True)
        for row in _sweep_mu(l_3yr, R_3yr, n_days_3yr, q=q_val):
            all_rows.append({"config": "3yr_Qsweep", **row})

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "results.csv", index=False)
    _make_plots(df, out_dir)
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name")
    args = parser.parse_args()
    run(args.experiment_name)
