import scipy.sparse as sp
import cvxpy as cp
import numpy as np

from mpc import mpc


def test_mpc():
    """
    Toy example to make sure the MPC is (syntactically) correct
    """

    # fmt: off
    data = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
    row = [0,0,1,1,1,2,3,3,3,4,5,5,5,6,6,8,8,9,11,12,1,4,2,3,4,3,4,6,8,5,10,11,12,7,8,9,13,10,12,13]
    col = [1,4,2,3,4,3,4,6,8,5,10,11,12,7,8,9,13,10,12,13,0,0,1,1,1,2,3,3,3,4,5,5,5,6,6,8,8,9,11,12]
    # fmt: on

    T = 10
    N = 14
    A = sp.coo_matrix((data, (row, col)), shape=(N, N))
    M = A.nnz // 2

    Q = np.array([100.0] * N)
    s = np.zeros(N)

    F = np.array([10.0] * M)
    C = np.array([10.0] * N)

    r = np.array([[1.0] * T] * N) * np.linspace(1, 0, T)
    r[0] = 0
    r[1] = 1

    g = np.array([[0.5] * T] * N)

    l = np.array([[1.0] * T] * N)

    c = mpc(s, Q, A, F, C, r, g, l)
    
    assert c[0]


if __name__ == "__main__":
    test_mpc()
