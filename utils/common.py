import numpy as np
from scipy.linalg import expm

def unpack_theta(theta):
    A = theta["A"]
    Gamma = theta["Gamma"]
    C = theta["C"]
    Sigma = theta["Sigma"]
    mu0 = theta["mu0"]
    V0 = theta["V0"]
    B = theta["B"]
    N = theta["N"]
    Ns = theta["Ns"]
    Nx = theta["Nx"]
    Nu = theta["Nu"]

    return A, Gamma, C, Sigma, mu0, V0, B, N, Ns, Nx, Nu

def reg_inv(mat, reg=1e-6):
    # return np.linalg.inv(mat + reg * np.eye(mat.shape[0]))
    return np.linalg.inv(mat)

def cont2disc_AQ(A_c, Q_c, dt):
    """
    Convert continuous A_c, Q_c to discrete A_d, Q_d over dt.
    """
    n = A_c.shape[0]
    # build block matrix
    M = np.zeros((2*n,2*n))
    M[:n,:n]   = -A_c
    M[:n,n:]   = Q_c
    M[n:,n:]   = A_c.T

    # exponentiate
    E = expm(M * dt)
    E12 = E[:n, n:]
    E22 = E[n:, n:]

    A_d = E22.T
    Q_d = A_d @ E12
    return A_d, Q_d
