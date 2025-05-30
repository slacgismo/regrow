import cvxpy as cp
import numpy as np
import scipy.sparse as sp


def mpc(
    s: np.array,
    Q: np.array,
    A: sp.sparray,
    F: np.array,
    C: np.array,
    r: np.array,
    g: np.array,
    l: np.array,
) -> np.array:
    """
    Args:
        s: battery SOC,             np.array of shape N
        A: adjacency matrix,        sp.sparray of shape N x N
        Q: battery capacity,        np.array of shape N
        F: line capacity,           np.array of shape M
        C: charge/discharge limit,  np.array of shape N
        r: renewable generation,    np.array of shape N x T
        g: fossil generation,       np.array of shape N x T
        l: load,                    np.array of shape N x T

    Returns:
        c: charge/discharge rate, np.array of shape N
        positive for charge, negative for discharge
    """
    I = adjacency_to_incidence(A)

    N = s.shape[0]
    T = r.shape[1]
    M = I.shape[1]

    c = cp.Variable((N, T))
    q = cp.Variable((N, T + 1), nonneg=True)
    r_tilde = cp.Variable((N, T), nonneg=True)
    f = cp.Variable((M, T))

    constraints = [
        r_tilde <= r,
        q[:, 0] == s,
        q[:, 1:] == q[:, :-1] + c,
        I @ f - c + r_tilde + g == l,
        q <= Q[:, None],
        cp.abs(f) <= F[:, None],
        cp.abs(c) <= C[:, None],
    ]

    tau_c = 1e-3
    tau_f = 1e-3
    regularizer = tau_c * cp.norm1(c) + tau_f * cp.sum_squares(f)
    objective = -cp.sum(q[:, -1]) + regularizer

    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=cp.CLARABEL, verbose=True)

    return c[:, 0].value


def adjacency_to_incidence(A: sp.sparray) -> sp.sparray:
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
