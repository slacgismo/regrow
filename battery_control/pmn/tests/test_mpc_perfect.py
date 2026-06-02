import pathlib

import numpy as np
import pytest
from controllers.data_utils import process_single_node_data
from controllers.mpc_perfect import run_mpc_perfect
from controllers.one_shot import load_one_shot_problem_data, make_one_shot

DATA_PATH = str(pathlib.Path(__file__).parent.parent.parent / "single_node_data.csv")

Q = 4.0
PARAMS = {
    "Q": Q,
    "B": Q / 2,
    "G": 1.0,
    "alpha": 1.25,
    "beta": 0.5,
    "lamb": 20.0,
    "mu": 1e-2,
    "gamma": 0.0,
    "q_init": Q / 2,
    "q_target": Q / 2,  # gamma is zero
    "round_trip_efficiency": 0.95,
    "monthly_soc_loss": 1,
}

l, R, _, _, _, _ = process_single_node_data(
    PARAMS["G"], data_path=DATA_PATH, data_start="2019-01-01", data_end="2019-01-14"
)

one_shot = make_one_shot(
    alpha=PARAMS["alpha"],
    beta=PARAMS["beta"],
    lamb=PARAMS["lamb"],
    mu=PARAMS["mu"],
    T=len(l),
    delta=1,
)
load_one_shot_problem_data(
    one_shot,
    l,
    R,
    G=PARAMS["G"],
    Q=PARAMS["Q"],
    B=PARAMS["B"],
    q0=PARAMS["q_init"],
    round_trip_efficiency=PARAMS["round_trip_efficiency"],
    monthly_soc_loss=PARAMS["monthly_soc_loss"],
)
one_shot.solve(solver="CLARABEL")
one_shot_sol = {k: v.value for k, v in one_shot.var_dict.items()}

T = len(l)
mpc_sol = run_mpc_perfect(l, R, H=T, **PARAMS)


@pytest.mark.parametrize("var", ["g", "s", "r", "c", "b", "q"])
def test_full_horizon_matches_one_shot(var):
    assert np.all(np.isclose(mpc_sol[var], one_shot_sol[var], atol=1e-1))


def objective(sol):
    return np.mean(
        PARAMS["lamb"] * sol["s"]
        + PARAMS["alpha"] * sol["g"]
        + PARAMS["beta"] * sol["g"] ** 2
        + PARAMS["mu"] * np.abs(sol["b"])
    )


def test_objective_terms_match_oneshot():
    tol = 1e-3
    assert np.isclose(np.sum(mpc_sol["s"]), np.sum(one_shot_sol["s"]), atol=tol)
    assert np.isclose(
        np.sum(PARAMS["alpha"] * mpc_sol["g"] + PARAMS["beta"] * mpc_sol["g"] ** 2),
        np.sum(PARAMS["alpha"] * one_shot_sol["g"] + PARAMS["beta"] * one_shot_sol["g"] ** 2),
        atol=tol,
    )
    assert np.isclose(np.sum(np.abs(mpc_sol["b"])), np.sum(np.abs(one_shot_sol["b"])), atol=tol)


def test_objective_matches_one_shot():
    assert np.isclose(objective(mpc_sol), objective(one_shot_sol), atol=1e-5)
