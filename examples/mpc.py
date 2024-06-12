import cvxpy as cp
import numpy as np
import scipy.sparse as sp


def mpc(
    s: np.array,
    Q: np.array,
    A: sp.spmatrix,
    F: np.array,
    C: np.array,
    r: np.array,
    g: np.array,
    l: np.array,
    R: np.array,
    kappa: np.array,
    verbose: bool = False
) -> np.array:
    """
    Args:
        s: battery SOC,             np.array of shape N
        A: incidence matrix,        sp.sparray of shape N x M
        Q: battery capacity,        np.array of shape N
        F: line capacity,           np.array of shape M
        C: charge/discharge limit,  np.array of shape N
        r: renewable generation,    np.array of shape N x T
        g: fossil generation,       np.array of shape N x T
        l: load,                    np.array of shape N x T
        R: line resistance,         np.array of shape M
        kappa: charge/discharge cost,   np.array of shape N

    Returns:
        c: charge/discharge rate, np.array of shape N
        positive for charge, negative for discharge
    """

    N, T = r.shape
    M = A.shape[1]
    Delta_T = 1

    c = cp.Variable((N, T))
    q = cp.Variable((N, T + 1), nonneg=True)
    r_curt = cp.Variable((N, T), nonneg=True)
    f = cp.Variable((M, T))

    constraints = [
        r_curt <= r,
        q[:, 0] == s,
        q[:, 1:] == q[:, :-1] + c * Delta_T,
        A @ f - c + r_curt + g == l,
        q <= Q[:, None],
        cp.abs(f) <= F[:, None],
        cp.abs(c) <= C[:, None],
    ]

    objective = -cp.sum(q[:, -1]) + cp.norm1(c) * Delta_T + cp.sum(R @ cp.square(f)) * Delta_T

    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=cp.CLARABEL, verbose=verbose)

    return c.value[:, 0]


def adjacency_to_incidence(A: sp.spmatrix) -> sp.spmatrix:
    """
    Args:
        A: adjacency matrix, sp.sparray of shape N x N

    Returns:
        I: incidence matrix, sp.sparray of shape N x M
    """

    assert (A + A.T).nnz == 0, "Expected undirected graph."
    assert np.all(A.diagonal() == 0), "Expected no self-loops."

    mask = A.row > A.col

    N = A.shape[0]
    M = np.sum(mask, dtype=int)

    data = np.concatenate([np.ones(M), -np.ones(M)])
    row = np.concatenate([A.row[mask], A.col[mask]])
    col = np.concatenate([np.arange(M), np.arange(M)])

    return sp.coo_matrix((data, (row, col)), shape=(N, M))
