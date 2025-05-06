import numpy as np
from scipy.linalg import expm, inv, logm

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

def disc2cont_AQ(A_d, Q_d, dt):
    """
    Reconstruct continuous‑time (A_c, Q_c) from discrete‑time (A_d, Q_d).

    Parameters
    ----------
    A_d : (n,n) ndarray
        Discrete‑time state‑transition matrix  (A_d = exp(A_c*dt)).
    Q_d : (n,n) ndarray
        Discrete‑time process‑noise covariance.
    dt  : float
        Sampling interval Δt used when the discrete pair was created.

    Returns
    -------
    A_c : (n,n) ndarray
        Continuous‑time drift matrix.
    Q_c : (n,n) ndarray
        Continuous‑time process‑noise spectral density.
    """
    n = A_d.shape[0]

    # --- build the inverse Van‑Loan block ----------------------------------
    #   E = [ A_d^{-1}           A_d^{-1} Q_d ]
    #       [     0                  A_d^T    ]
    E = np.zeros((2 * n, 2 * n))
    A_d_inv = inv(A_d)

    E[:n, :n]   = A_d_inv
    E[:n, n:]   = A_d_inv @ Q_d
    E[n:, n:]   = A_d.T

    # --- matrix logarithm ---------------------------------------------------
    # log(E) = [ -A_c   Q_c ]
    #          [  0     A_c^T ]
    M = logm(E) / dt

    # numerical noise from logm() can leave tiny imaginary parts
    M = np.real_if_close(M, tol=1000)

    # --- extract continuous matrices ---------------------------------------
    A_c = -M[:n, :n]
    Q_c =  M[:n, n:]

    return A_c, Q_c

def cont2disc(theta, dt):
    # Continuous→discrete for A, Gamma, and B
    A_c = theta.A
    Gamma_c = theta.Gamma
    B_c = theta.B

    # Discretize A and Gamma
    A_d, Gamma_d = cont2disc_AQ(A_c, Gamma_c, dt)

    # Discretize B via block‐matrix exponential:
    # B_d = ∫₀ᵈᵗ e^{A_c τ} B_c dτ
    n, m = B_c.shape
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A_c
    M[:n, n:] = B_c
    E = expm(M * dt)
    B_d = E[:n, n:]

    # Assign back to theta
    theta.A = A_d
    theta.Gamma = Gamma_d
    theta.B = B_d
    return theta

def disc2cont(theta, dt):
    A_d = theta.A
    Gamma_d = theta.Gamma
    B_d = theta.B

    n, m = B_d.shape

    # Recover continuous A and Gamma
    A_c, Gamma_c = disc2cont_AQ(A_d, Gamma_d, dt)

    # Recover continuous B using block logm
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A_d
    M[:n, n:] = B_d
    M[n:, n:] = np.eye(m)

    M_log = logm(M) / dt
    A_c_from_block = M_log[:n, :n]
    B_c = M_log[:n, n:]

    # Assign back to theta
    theta.A = A_c
    theta.Gamma = Gamma_c
    theta.B = B_c
    return theta
