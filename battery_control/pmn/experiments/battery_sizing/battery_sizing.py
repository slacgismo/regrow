from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import product

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from controllers.metrics import get_metrics_of_interest
from controllers.mpc_perfect import run_mpc_perfect
from tqdm import tqdm


def _run_pair(fixed_kwargs, pair):
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
    q_init=None,
    efficiency=0.98,
    soc_loss=0,
    delta=1,
    solver="CLARABEL",
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
        efficiency=efficiency,
        soc_loss=soc_loss,
        delta=delta,
        solver=solver,
        disable_progress_bar=True,
    )

    pairs = list(product(gamma_list, q_target_list))
    run = partial(_run_pair, fixed_kwargs)

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
    best_row = df.loc[best_idx]
    return best_row
