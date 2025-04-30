import numpy as np

def unpack_theta(theta):
    A = theta["A"]
    Gamma = theta["Gamma"]
    C = theta["C"]
    Sigma = theta["Sigma"]
    mu0 = theta["mu0"]
    V0 = theta["V0"]
    B = theta["B"]
    N = theta["N"]
    Nx = theta["Nx"]
    Nu = theta["Nu"]

    return A, Gamma, C, Sigma, mu0, V0, B, N, Nx, Nu

def reg_inv(mat, reg=1e-6):
    return np.linalg.inv(mat + reg * np.eye(mat.shape[0]))
