import argparse
import json
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from controllers.data_utils import build_lsw_df, sample_stress_event
from controllers.metrics import get_metrics_of_interest
from controllers.mpc_perfect import run_mpc_perfect
from controllers.one_shot import load_one_shot_problem_data, make_one_shot

G = 1
BAT_HOURS = 4
ROUND_TRIP_EFFICIENCY = 0.95
MONTHLY_SOC_LOSS = 1
ALPHA = 1.25
BETA = 0.5
LAMB = 20.0
MU = 0.01
Q = 10
Q_INIT = Q / 2

GAMMA_LIST = np.logspace(-4, -1, num=4)
Q_TARGET_LIST = [0.25, 0.5, 0.75, 1.0]
H_LIST = [24, 48, 72]

DATA_START = "2018"
DATA_END = "2020"

N_MC_EVENTS = 20
SCALING_RATIO = 2 / 3
ENERGY_RANGE_GWH = (5.0, 200.0)
DURATION_RANGE_DAYS = (1, 14)
RNG_SEED = 42

MAX_WORKERS = 8


def _run_oneshot(l, R, event_mask, delta):
    T = len(l)
    problem = make_one_shot(alpha=ALPHA, beta=BETA, lamb=LAMB, mu=MU, T=T, delta=delta)
    load_one_shot_problem_data(
        problem, l=l, R=R, G=G, Q=Q, B=Q / BAT_HOURS, q0=Q_INIT,
        round_trip_efficiency=ROUND_TRIP_EFFICIENCY, monthly_soc_loss=MONTHLY_SOC_LOSS,
    )
    problem.solve()
    vd = problem.var_dict
    return get_metrics_of_interest(
        vd["s"].value, vd["g"].value, vd["b"].value, vd["c"].value,
        LAMB, ALPHA, BETA, MU, delta=delta, stress_mask=event_mask,
    )


def _run_one(args):
    gamma, q_target, H, l, R, event_mask, event_id, is_regression, oneshot_metrics, fixed_kwargs = args
    base = {"gamma": gamma, "q_target": q_target, "H": H, "event_id": event_id, "is_regression": is_regression}
    delta = fixed_kwargs["delta"]
    try:
        sol = run_mpc_perfect(gamma=gamma, q_target=q_target, H=H, l=l, R=R, **fixed_kwargs)
    except Exception as e:
        print(f"ERROR gamma={gamma:.0e} q_target={q_target} H={H} event_id={event_id}: {e}", flush=True)
        mpc_metrics = {f"mpc_{k}": float("nan") for k in oneshot_metrics}
    else:
        raw = get_metrics_of_interest(
            sol["s"], sol["g"], sol["b"], sol["c"], LAMB, ALPHA, BETA, MU, delta=delta, stress_mask=event_mask,
        )
        mpc_metrics = {f"mpc_{k}": float(v) for k, v in raw.items()}
    return {**base, **mpc_metrics, **{f"oneshot_{k}": float(v) for k, v in oneshot_metrics.items()}}


def run(experiment_name):
    data_path = str(pathlib.Path(__file__).parent.parent.parent.parent / "single_node_data.csv")
    out_dir = pathlib.Path(__file__).parent / experiment_name
    out_dir.mkdir(exist_ok=True)

    config = {
        "data": {"data_start": DATA_START, "data_end": DATA_END},
        "fixed_params": {
            "G": G, "Q": Q, "BAT_HOURS": BAT_HOURS,
            "ROUND_TRIP_EFFICIENCY": ROUND_TRIP_EFFICIENCY, "MONTHLY_SOC_LOSS": MONTHLY_SOC_LOSS,
            "ALPHA": ALPHA, "BETA": BETA, "LAMB": LAMB, "MU": MU,
        },
        "sweep": {
            "gamma_list": GAMMA_LIST.tolist(),
            "q_target_list": Q_TARGET_LIST,
            "H_list": H_LIST,
        },
        "stress_events": {
            "N_MC_EVENTS": N_MC_EVENTS,
            "SCALING_RATIO": SCALING_RATIO,
            "ENERGY_RANGE_GWH": list(ENERGY_RANGE_GWH),
            "DURATION_RANGE_DAYS": list(DURATION_RANGE_DAYS),
            "RNG_SEED": RNG_SEED,
        },
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    print("loading data...", flush=True)
    lsw_base = build_lsw_df(data_path=data_path, data_start=DATA_START, data_end=DATA_END)
    l_base = lsw_base["l"].to_numpy()
    R_base = (lsw_base["s"] + lsw_base["w"]).to_numpy()
    delta = (lsw_base.index[1] - lsw_base.index[0]).total_seconds() / 3600

    print("generating stress events...", flush=True)
    rng = np.random.default_rng(RNG_SEED)
    event_meta = []
    stress_events = []
    for i in range(N_MC_EVENTS):
        lsw_ev, ev_start, ev_duration, ev_energy, _ = sample_stress_event(
            lsw_base, scaling_ratio=SCALING_RATIO,
            duration_range_days=DURATION_RANGE_DAYS, energy_range_gwh=ENERGY_RANGE_GWH,
            rng=rng,
        )
        ev_end = ev_start + ev_duration
        ev_mask = lsw_base.index.isin(lsw_base.loc[ev_start:ev_end].index)
        stress_events.append((i, lsw_ev["l"].to_numpy(), (lsw_ev["s"] + lsw_ev["w"]).to_numpy(), ev_mask))
        event_meta.append({"event_id": i, "start": str(ev_start.date()), "duration_days": ev_duration.days, "energy_gwh": ev_energy})
    pd.DataFrame(event_meta).to_csv(out_dir / "events.csv", index=False)

    print("running one-shot solves...", flush=True)
    oneshot_per_event = {-1: _run_oneshot(l_base, R_base, None, delta=delta)}
    for event_id, l_ev, R_ev, ev_mask in stress_events:
        oneshot_per_event[event_id] = _run_oneshot(l_ev, R_ev, ev_mask, delta=delta)
        print(f"  one-shot {event_id + 1}/{N_MC_EVENTS}", flush=True)

    fixed_kwargs = dict(
        G=G, Q=Q, B=Q / BAT_HOURS, alpha=ALPHA, beta=BETA, lamb=LAMB, mu=MU,
        q_init=Q_INIT, round_trip_efficiency=ROUND_TRIP_EFFICIENCY,
        monthly_soc_loss=MONTHLY_SOC_LOSS, delta=delta, disable_progress_bar=True,
    )

    all_events = [(-1, l_base, R_base, None, True)] + [
        (eid, l_ev, R_ev, ev_mask, False) for eid, l_ev, R_ev, ev_mask in stress_events
    ]
    n_inner = len(GAMMA_LIST) * len(Q_TARGET_LIST) * len(H_LIST)
    total = len(all_events) * n_inner
    print(f"running {total} MPC solves ({len(all_events)} events x {n_inner} param combos) with {MAX_WORKERS} workers...", flush=True)

    all_rows = []
    completed = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for event_idx, (event_id, l_ev, R_ev, ev_mask, is_regression) in enumerate(all_events):
            oneshot_metrics = oneshot_per_event[event_id]
            tasks = [
                (gamma, q_target, H, l_ev, R_ev, ev_mask, event_id, is_regression, oneshot_metrics, fixed_kwargs)
                for gamma, q_target, H in product(GAMMA_LIST, Q_TARGET_LIST, H_LIST)
            ]
            futures = {pool.submit(_run_one, t): t for t in tasks}
            for future in as_completed(futures):
                row = future.result()
                all_rows.append(row)
                completed += 1
                pd.DataFrame(all_rows).to_csv(out_dir / "results.csv", index=False)
            label = "regression" if is_regression else f"stress event {event_id}"
            print(f"  [{event_idx + 1}/{len(all_events)}] {label} complete ({completed}/{total} total)", flush=True)

    print(f"done. saved to {out_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name")
    args = parser.parse_args()
    run(args.experiment_name)
