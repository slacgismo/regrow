import argparse
import json
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import product

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from controllers.constraints import validate_battery_dynamics
from controllers.data_utils import process_single_node_data
from controllers.metrics import core_objective
from controllers.mpc_perfect import run_mpc_perfect

G = 1
BAT_HOURS = 4
EFFICIENCY = 0.98
SOC_LOSS = 1e-5
ALPHA = 1.25
BETA = 0.5
LAMB = 20.0
Q = 10
Q_INIT = Q / 2

MU_LIST = np.logspace(-6, 0, num=7)
H_LIST = [24, 48, 72]
GAMMA_LIST = np.logspace(-4, -1, num=4)
Q_TARGET_LIST = [0.5, 0.75, 1.0]

MAX_WORKERS = 4


def _run_one(args):
    mu, H, gamma, q_target, fixed_kwargs, n_days = args
    base = {"mu": mu, "H": H, "gamma": gamma, "q_target": q_target}
    try:
        sol = run_mpc_perfect(mu=mu, H=H, gamma=gamma, q_target=q_target, **fixed_kwargs)
    except Exception as e:
        print(f"  ERROR mu={mu:.0e} H={H} gamma={gamma:.0e} q_target={q_target}: {e}", flush=True)
        return {**base, "valid": False, "core_objective": float("nan"), "throughput_per_day": float("nan"), "non_battery_cost_per_day": float("nan")}
    valid = validate_battery_dynamics(
        q=sol["q"],
        b=sol["b"],
        b_out=sol["b_out"],
        b_in=sol["b_in"],
        Q=Q,
        B=Q / BAT_HOURS,
        q0=Q_INIT,
        charge_efficiency=EFFICIENCY,
        discharge_efficiency=EFFICIENCY,
        soc_loss=SOC_LOSS,
        delta=1,
    )
    return {
        **base,
        "valid": valid,
        "core_objective": float(core_objective(sol["s"], sol["g"], sol["b"], LAMB, ALPHA, BETA, mu).value),
        "throughput_per_day": float(np.sum(np.abs(sol["b"])) / n_days),
        "non_battery_cost_per_day": float(np.sum(LAMB * sol["s"] + ALPHA * sol["g"] + BETA * sol["g"] ** 2) / n_days),
    }


def _sweep(l, R, n_days):
    fixed_kwargs = dict(
        l=l,
        R=R,
        G=G,
        Q=Q,
        B=Q / BAT_HOURS,
        alpha=ALPHA,
        beta=BETA,
        lamb=LAMB,
        q_init=Q_INIT,
        efficiency=EFFICIENCY,
        soc_loss=SOC_LOSS,
        disable_progress_bar=True,
    )
    args = [
        (mu, H, gamma, q_target, fixed_kwargs, n_days)
        for mu, H, gamma, q_target in product(MU_LIST, H_LIST, GAMMA_LIST, Q_TARGET_LIST)
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for i, row in enumerate(pool.map(_run_one, args), 1):
            print(
                f"  [{i}/{len(args)}] mu={row['mu']:.0e} H={row['H']} gamma={row['gamma']:.0e} q_target={row['q_target']} valid={row['valid']}",
                flush=True,
            )
            rows.append(row)
    return rows


def _plot_group(ax, df, names, labels=None, col="H"):
    for name, label in zip(names, labels or names):
        sub = df[df[col] == name].sort_values("mu")
        ax.plot(sub["throughput_per_day"], sub["non_battery_cost_per_day"], marker="o", label=label)
    ax.set_xlabel("battery throughput [GWh/day]")
    ax.set_ylabel("operational cost (load shed + generation) [/day]")
    ax.legend()


def _plot_group_vs_mu(ax, df, names, labels=None, col="H"):
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
    H_values = sorted(df["H"].unique())
    H_labels = [f"H={H}" for H in H_values]
    best = df[df["valid"]].loc[
        lambda d: d.groupby(["mu", "H"])["core_objective"].transform("min") == d["core_objective"]
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    _plot_group(ax, best, H_values, labels=H_labels)
    ax.set_title("mu pareto: mpc perfect (3yr)")
    _save(fig, out_dir / "pareto.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    _plot_group_vs_mu(ax, best, H_values, labels=H_labels)
    ax.set_title("mu vs cost: mpc perfect (3yr)")
    _save(fig, out_dir / "mu_cost.png")

    fig, axes = plt.subplots(1, len(H_values), figsize=(5 * len(H_values), 4))
    log_mu_min, log_mu_max = np.log10(MU_LIST.min()), np.log10(MU_LIST.max())
    for ax, H in zip(axes, H_values):
        sub = df[df["H"] == H]
        pivot = (
            sub.groupby(["gamma", "q_target"])
            .apply(lambda g: g[g["valid"]]["mu"].min() if g["valid"].any() else np.nan)
            .unstack("q_target")
        )
        log_pivot = np.log10(pivot.values.astype(float))
        im = ax.imshow(log_pivot, aspect="auto", cmap="RdYlBu_r", vmin=log_mu_min, vmax=log_mu_max)
        for i in range(log_pivot.shape[0]):
            for j in range(log_pivot.shape[1]):
                val = pivot.values[i, j]
                ax.text(j, i, f"{val:.0e}" if not np.isnan(val) else "none", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(len(pivot.columns)), labels=[str(v) for v in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), labels=[f"{v:.0e}" for v in pivot.index])
        ax.set_xlabel("q_target")
        ax.set_ylabel("gamma")
        ax.set_title(f"H={H}: smallest valid mu")
        fig.colorbar(im, ax=ax, label="log10(mu)")
    _save(fig, out_dir / "validity.png")


def run(experiment_name):
    data_path = str(pathlib.Path(__file__).parent.parent.parent.parent / "single_node_data.csv")
    out_dir = pathlib.Path(__file__).parent / experiment_name
    out_dir.mkdir(exist_ok=True)

    config = {
        "data": {"data_start": "2018", "data_end": "2020"},
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
        "sweep": {
            "mu_list": MU_LIST.tolist(),
            "H_list": H_LIST,
            "gamma_list": GAMMA_LIST.tolist(),
            "q_target_list": Q_TARGET_LIST,
        },
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    l, R, _, tidx, *_ = process_single_node_data(
        G,
        data_path=data_path,
        data_start="2018",
        data_end="2020",
        add_event=False,
    )
    n_days = len(tidx) * (tidx[1] - tidx[0]) / pd.Timedelta("1D")

    print(f"total solves: {len(MU_LIST) * len(H_LIST) * len(GAMMA_LIST) * len(Q_TARGET_LIST)}", flush=True)
    df = pd.DataFrame(_sweep(l, R, n_days))
    df.to_csv(out_dir / "results.csv", index=False)
    _make_plots(df, out_dir)
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name")
    args = parser.parse_args()
    run(args.experiment_name)
