import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import product

import numpy as np
import pandas as pd
from controllers.data_utils import build_lsw_df, sample_stress_event
from controllers.metrics import get_metrics_of_interest
from controllers.mpc_perfect import run_mpc_perfect
from controllers.one_shot import load_one_shot_problem_data, make_one_shot
from tqdm import tqdm


def _run_pair(fixed_kwargs, stress_mask, H, pair):
    gamma, q_target = pair
    sol = run_mpc_perfect(H=H, gamma=gamma, q_target=q_target, **fixed_kwargs)
    metrics = get_metrics_of_interest(
        s=sol["s"],
        g=sol["g"],
        b=sol["b"],
        c=sol["c"],
        lamb=fixed_kwargs["lamb"],
        alpha=fixed_kwargs["alpha"],
        beta=fixed_kwargs["beta"],
        mu=fixed_kwargs["mu"],
        delta=fixed_kwargs["delta"],
        stress_mask=stress_mask,
    )
    return {"H": H, "gamma": gamma, "q_target": q_target, **metrics}


def _solve_one_shot(problem, l, R, G, Q, B, q_init, efficiency, soc_loss, alpha, beta, lamb, mu, delta, stress_mask):
    load_one_shot_problem_data(
        problem,
        l=l,
        R=R,
        G=G,
        Q=Q,
        B=B,
        q0=q_init,
        round_trip_efficiency=efficiency,
        monthly_soc_loss=soc_loss,
    )
    problem.solve(solver="CLARABEL")
    sol = problem.var_dict
    return get_metrics_of_interest(
        s=sol["s"].value,
        g=sol["g"].value,
        b=sol["b"].value,
        c=sol["c"].value,
        lamb=lamb,
        alpha=alpha,
        beta=beta,
        mu=mu,
        delta=delta,
        stress_mask=stress_mask,
    )


def run_sensitivity_experiment(experiment_name):
    DATA_START = "2019"
    DATA_END = "2019"
    G = 1
    Q = 16
    BAT_HOURS = 4
    EFFICIENCY = 0.95
    SOC_LOSS = 2
    ALPHA = 1.25
    BETA = 0.5
    LAMB = 20.0
    MU = 0.01
    DELTA = 1

    H_LIST = [24, 48, 72]
    GAMMA_LIST = np.logspace(-4, -1, num=4).tolist()
    Q_TARGET_LIST = [0.5, 0.75, 1.0]

    N_SAMPLES = 20
    SCALING_RATIO = 2 / 3
    DURATION_RANGE_DAYS = [1, 5]
    SHORTFALL_RANGE_GWH = [Q / 2, 1.5 * Q]
    START_BUFFER_DAYS = 7  # no generated stress events during this period
    BASE_SEED = 0
    MAX_WORKERS = 6

    config = {
        "data": {"data_start": DATA_START, "data_end": DATA_END},
        "fixed_params": {
            "G": G,
            "Q": Q,
            "bat_hours": BAT_HOURS,
            "efficiency": EFFICIENCY,
            "soc_loss": SOC_LOSS,
            "alpha": ALPHA,
            "beta": BETA,
            "lamb": LAMB,
            "mu": MU,
            "delta": DELTA,
        },
        "sweep": {
            "H_list": H_LIST,
            "gamma_list": GAMMA_LIST,
            "q_target_list": Q_TARGET_LIST,
        },
        "stress_sampling": {
            "n_samples": N_SAMPLES,
            "scaling_ratio": SCALING_RATIO,
            "duration_range_days": DURATION_RANGE_DAYS,
            "shortfall_range_gwh": SHORTFALL_RANGE_GWH,
            "start_buffer_days": START_BUFFER_DAYS,
            "base_seed": BASE_SEED,
        },
    }

    data_path = str(pathlib.Path(__file__).parent.parent.parent.parent / "single_node_data.csv")
    base_lsw_df = build_lsw_df(data_path=data_path, data_start=DATA_START, data_end=DATA_END)
    delta_hours = (base_lsw_df.index[1] - base_lsw_df.index[0]).total_seconds() / 3600
    l_base = base_lsw_df["l"].to_numpy()
    R_base = (base_lsw_df["s"] + base_lsw_df["w"]).to_numpy()
    B = Q / BAT_HOURS
    q_init = Q / 2

    out_dir = pathlib.Path(__file__).parent / experiment_name
    out_dir.mkdir(exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    pairs = list(product(GAMMA_LIST, Q_TARGET_LIST))

    one_shot_problem = make_one_shot(alpha=ALPHA, beta=BETA, lamb=LAMB, mu=MU, T=len(l_base), delta=DELTA)
    one_shot_kwargs = dict(
        G=G,
        Q=Q,
        B=B,
        q_init=q_init,
        efficiency=EFFICIENCY,
        soc_loss=SOC_LOSS,
        alpha=ALPHA,
        beta=BETA,
        lamb=LAMB,
        mu=MU,
        delta=DELTA,
    )

    # --- normal data ---
    print("Running normal data...", flush=True)

    os_metrics = _solve_one_shot(one_shot_problem, l=l_base, R=R_base, stress_mask=None, **one_shot_kwargs)
    pd.DataFrame([os_metrics]).to_csv(out_dir / "one_shot_normal.csv", index=False)

    mpc_kwargs_base = dict(
        l=l_base,
        R=R_base,
        G=G,
        Q=Q,
        B=B,
        alpha=ALPHA,
        beta=BETA,
        lamb=LAMB,
        mu=MU,
        q_init=q_init,
        round_trip_efficiency=EFFICIENCY,
        monthly_soc_loss=SOC_LOSS,
        delta=DELTA,
        solver="CLARABEL",
        disable_progress_bar=True,
    )
    normal_rows = []
    for H in H_LIST:
        run = partial(_run_pair, mpc_kwargs_base, None, H)
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
            normal_rows.extend(tqdm(pool.map(run, pairs), total=len(pairs), desc=f"normal MPC H={H}"))
    pd.DataFrame(normal_rows).to_csv(out_dir / "results_normal.csv", index=False)

    # --- stress data ---
    results_stress_path = out_dir / "results_stress.csv"
    one_shot_stress_path = out_dir / "one_shot_stress.csv"
    all_stress_rows = []
    all_os_stress_rows = []

    for sample_idx in range(N_SAMPLES):
        print(f"[{sample_idx + 1}/{N_SAMPLES}] sampling stress event...", flush=True)
        rng = np.random.default_rng(BASE_SEED + sample_idx)
        lsw_stressed, event_start, event_duration, event_shortfall = sample_stress_event(
            base_lsw_df,
            G=G,
            scaling_ratio=SCALING_RATIO,
            duration_range_days=DURATION_RANGE_DAYS,
            shortfall_range_gwh=SHORTFALL_RANGE_GWH,
            start_buffer=pd.Timedelta(days=START_BUFFER_DAYS),
            rng=rng,
        )
        l_s = lsw_stressed["l"].to_numpy()
        R_s = (lsw_stressed["s"] + lsw_stressed["w"]).to_numpy()
        stress_mask = (lsw_stressed.index >= event_start) & (lsw_stressed.index < event_start + event_duration)

        period = base_lsw_df.index[1] - base_lsw_df.index[0]
        event_slice = base_lsw_df.loc[event_start : event_start + event_duration - period]
        baseline_shortfall = delta_hours * float(
            np.sum(np.maximum(event_slice["l"] - (event_slice["s"] + event_slice["w"]) - G, 0))
        )

        event_meta = {
            "sample_idx": sample_idx,
            "event_start": str(event_start.date()),
            "event_duration_days": int(event_duration.days),
            "baseline_shortfall_gwh": baseline_shortfall,
            "added_shortfall_gwh": float(event_shortfall),
        }

        os_metrics = _solve_one_shot(one_shot_problem, l=l_s, R=R_s, stress_mask=stress_mask, **one_shot_kwargs)
        all_os_stress_rows.append({**event_meta, **os_metrics})
        pd.DataFrame(all_os_stress_rows).to_csv(one_shot_stress_path, index=False)

        mpc_kwargs_s = dict(
            l=l_s,
            R=R_s,
            G=G,
            Q=Q,
            B=B,
            alpha=ALPHA,
            beta=BETA,
            lamb=LAMB,
            mu=MU,
            q_init=q_init,
            round_trip_efficiency=EFFICIENCY,
            monthly_soc_loss=SOC_LOSS,
            delta=DELTA,
            solver="CLARABEL",
            disable_progress_bar=True,
        )
        for H in H_LIST:
            run = partial(_run_pair, mpc_kwargs_s, stress_mask, H)
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
                h_rows = list(
                    tqdm(pool.map(run, pairs), total=len(pairs), desc=f"stress MPC sample={sample_idx + 1} H={H}")
                )
            for row in h_rows:
                all_stress_rows.append({**event_meta, **row})
            pd.DataFrame(all_stress_rows).to_csv(results_stress_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name")
    args = parser.parse_args()
    run_sensitivity_experiment(args.experiment_name)
