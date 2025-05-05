import numpy as np
import matplotlib.pyplot as plt
from utils.plotting import plot_data, plot_theta_diffs, plot_loss
from utils.common import unpack_theta, reg_inv, cont2disc_AQ
from vehicle import SimpleLinearModel
from policies import NoControl
from environment import SimpleEnv
from filterpy.kalman import KalmanFilter
    

## --- Kalman filter ---
def P(V_n, A, Gamma): # Predicted state covariance
    return A @ V_n @ A.T + Gamma

def K(P_nm1, C, Sigma): # Kalman gain
    return P_nm1 @ C.T @ reg_inv(C @ P_nm1 @ C.T + Sigma)

def kalman_filter(theta, X):

    A = theta.A
    Gamma = theta.Gamma
    C = theta.C
    Sigma = theta.Sigma
    mu0 = theta.mu0
    V0 = theta.V0
    N = theta.N
    Ns = theta.Ns

    V = np.zeros((N, Ns, Ns))
    mu = np.zeros((N, Ns))
    K1 = K(V0, C, Sigma)

    # V[0] = V0
    # mu[0] = mu0
    V[0] = (np.eye(Ns) - K1 @ C) @ V0
    mu[0] = mu0 + K1 @ (X[0] - C @ mu0)
    
    for n in range(1, N):
        Pnm1 = P(V[n-1], A, Gamma)
        Kn = K(Pnm1, C, Sigma)

        mu[n] = A @ mu[n-1] + Kn @ (X[n] - C @ A @ mu[n-1])
        V[n] = (np.eye(Ns) - Kn @ C) @ Pnm1

    return mu, V

def kalman_filter_filterpy(theta, X):
    kf = KalmanFilter(dim_x=theta.Ns, dim_z=theta.Nx)
    kf.x = theta.mu0
    kf.P = theta.V0
    kf.F = theta.A
    kf.Q = theta.Gamma
    kf.H = theta.C
    kf.R = theta.Sigma
    mu, V, _, _ = kf.batch_filter(X)
    mu_hat, V_hat, _, _ = kf.rts_smoother(mu, V)

    return mu_hat, V_hat, mu, V

## --- Kalman smoother ---
def J(P_n, V_n, A):
    return V_n @ A.T @ reg_inv(P_n)

def kalman_smoother(theta, X, mu, V):
    A = theta.A
    Gamma = theta.Gamma
    N = theta.N
    Ns = theta.Ns
    mu_hat_rev = np.empty((0, Ns))
    mu_hat_rev = np.vstack((mu_hat_rev, mu[N-1]))
    V_hat_rev = np.empty((0, Ns, Ns)) # Reverse order of V
    V_hat_rev = np.vstack((V_hat_rev, [V[N-1]]))   

    for n in range(1, N):
        mu_n = mu[N-n-1]
        mu_hat_nm1 = mu_hat_rev[n-1]
        Vn = V[N-n-1]
        Pn = P(Vn, A, Gamma)
        Jn = J(Pn, Vn, A)
        mu_hat_rev = np.vstack((mu_hat_rev, mu_n + Jn @ (mu_hat_nm1 - A @ mu_n)))
        V_hat_rev = np.vstack((V_hat_rev, [Vn + Jn @ (V_hat_rev[n-1] - Pn) @ Jn.T]))

    V_hat = np.flip(V_hat_rev, axis=0)
    mu_hat = np.flip(mu_hat_rev, axis=0)

    return mu_hat, V_hat

## --- Maximization ---
def get_E1(mu_hat):
    return mu_hat

def get_E2_E3(theta, V, mu_hat, V_hat):
    A = theta.A
    Gamma = theta.Gamma
    N = theta.N
    Ns = theta.Ns

    E2 = np.empty((0,Ns,Ns))
    E3 = np.empty((0,Ns,Ns))
    for n in range(1,N):
        P_n_minus1 = P(V[n-1], A, Gamma)
        J_n_minus1 = J(P_n_minus1, V[n-1], A)
        mu_hat_n_T = np.array([mu_hat[n]]).T
        mu_hat_n_minus1 = np.array([mu_hat[n-1]])
        E2_n = V_hat[n] @ J_n_minus1.T + mu_hat_n_T @ mu_hat_n_minus1
        E2 = np.vstack((E2, [E2_n]))
        E3_n = E2_n.T
        E3 = np.vstack((E3, [E3_n]))
    return E2, E3

def get_E4(theta, mu_hat, V_hat):
    N = theta.N
    Ns = theta.Ns

    E4 = np.empty((0,Ns,Ns))
    for n in range(N):
        mu_hat_n = np.array([mu_hat[n]])
        mu_hat_n_T = mu_hat_n.T
        E4_n = V_hat[n] + mu_hat_n_T @ mu_hat_n
        E4 = np.vstack((E4, [E4_n]))
    return E4

def get_mu0_new(theta, E1, multi=False):
    Nk = theta.Nk
    if not multi:
        return E1[0]
    else:
        return np.sum(E1[:,0], axis=0) / Nk

def get_V0_new(theta, E1, E4, multi=False):
    Nk = theta.Nk
    if not multi:
        return E4[0] - E1[0] @ E1[0].T
    else:
        E1_0_hat = np.sum(E1[:,0], axis=0) / Nk
        return np.sum(E4[:,0], axis=0) / Nk - E1_0_hat @ E1_0_hat.T + np.sum([np.outer(e, e) for e in (E1[:,0] - E1_0_hat)], axis=0) / Nk

def get_A_new(theta, E2, E4, multi=False):
    N = theta.N
    Nk = theta.Nk
    Ns = theta.Ns
    if not multi:
        return np.sum(E2, axis=0) @ np.linalg.inv(np.sum(E4[:N-1], axis=0))
    else:
        E2_sum = np.zeros((Ns, Ns))
        E4_sum = np.zeros((Ns, Ns))
        for k in range(Nk):
            E2_sum += np.sum(E2[k], axis=0)
            E4_sum += np.sum(E4[k, :N-1], axis=0)
        return (E2_sum @ np.linalg.inv(E4_sum))

def get_Gamma_new(theta, E2, E3, E4, A_new, multi=False):
    Ns = theta.Ns
    N = theta.N
    Nk = theta.Nk
    if not multi:
        elems = np.empty((0,Ns,Ns))
        for n in range(1, N):
            elems_n = (E4[n] - A_new @ E3[n-1] - E2[n-1] @ A_new.T + A_new @ E4[n-1] @ A_new.T)
            elems = np.vstack((elems, [elems_n]))
        return np.sum(elems, axis=0) / (N-1)
    else:
        Gamma_k = np.zeros((Ns,Ns))
        for k in range(Nk):
            elems = np.empty((0,Ns,Ns))
            for n in range(1, N):
                elems_n = E4[k, n] - A_new @ E3[k, n-1] - E2[k, n-1] @ A_new.T + A_new @ E4[k, n-1] @ A_new.T
                elems = np.vstack((elems, [elems_n]))
            Gamma_k += np.sum(elems, axis=0)
        return Gamma_k / (Nk*(N-1))


def get_C_new_former(theta, E1, X_k):
    Ns = theta.Ns
    Nx = theta.Nx
    N = theta.N
    elems = np.empty((0,Nx,Ns))
    for n in range(N):
        x_n_T = np.array([X_k[n]]).T
        E1_n = np.array([E1[n]])
        elems_n = x_n_T @ E1_n
        elems = np.vstack((elems, [elems_n]))
    return np.sum(elems, axis=0)

def get_C_new(theta, E1, E4, X, multi=False):
    if not multi:
        return get_C_new_former(theta, E1, X) @ np.linalg.inv(np.sum(E4, axis=0))
    else:
        Nk = theta.Nk
        Nx = theta.Nx
        Ns = theta.Ns
        C_sum_1 = np.zeros((Nx,Ns))
        for k in range(Nk):
            C_sum_1 += get_C_new_former(theta, E1[k], X[:,:,k])
        C_sum_2 = np.zeros((Ns,Ns))
        for k in range(Nk):
            C_sum_2 += np.sum(E4[k,:], axis=0)
        return C_sum_1 @ np.linalg.inv(C_sum_2)


def get_Sigma_new(theta, E1, E4, C_new, X, multi=False):
    Ns = theta.Ns
    Nx = theta.Nx
    N = theta.N
    Nk = theta.Nk
    if not multi:
        elems = np.empty((0,Nx,Nx))
        for n in range(N):
            x_n = np.array([X[n]])
            x_n_T = x_n.T
            E1_n = np.array([E1[n]])
            E1_n_T = E1_n.T
            elem_n = x_n_T @ x_n - C_new @ E1_n_T @ x_n - x_n_T @ E1_n @ C_new.T + C_new @ E4[n] @ C_new.T
            elems = np.vstack((elems, [elem_n]))
        return np.sum(elems, axis=0) / N
    else:
        Sigma_sum = np.zeros((Nx,Nx))
        for k in range(Nk):
            elems = np.empty((0,Nx,Nx))
            X_k = X[:,:,k]
            for n in range(N):
                x_n = np.array([X_k[n]])
                x_n_T = x_n.T
                E1_n = np.array([E1[k, n]])
                E1_n_T = E1_n.T
                elem_n = x_n_T @ x_n - C_new @ E1_n_T @ x_n - x_n_T @ E1_n @ C_new.T + C_new @ E4[k, n] @ C_new.T
                elems = np.vstack((elems, [elem_n]))
            Sigma_sum += np.sum(elems, axis=0)
        return Sigma_sum / (Nk*N)

def maximization(theta, X, E1, E2, E3, E4):
    mu0 = get_mu0_new(theta, E1)
    V0 = get_V0_new(theta, E1, E4)

    A_new = get_A_new(theta, E2, E4)
    Gamma_new = get_Gamma_new(theta, E2, E3, E4, A_new)

    C_new = get_C_new(theta, E1, E4, X)
    Sigma_new = get_Sigma_new(theta, E1, E4, C_new, X)

    return mu0, V0, A_new, Gamma_new, C_new, Sigma_new

def maximization_multi(theta, X, E1, E2, E3, E4):

    theta_new = theta.copy()
    theta_new.mu0 = get_mu0_new(theta, E1, multi=True)
    theta_new.V0 = get_V0_new(theta, E1, E4, multi=True)

    A_new = get_A_new(theta, E2, E4, multi=True)
    theta_new.A = A_new
    theta_new.Gamma = get_Gamma_new(theta, E2, E3, E4, A_new, multi=True)

    C_new = get_C_new(theta, E1, E4, X, multi=True)
    theta_new.C = C_new
    theta_new.Sigma = get_Sigma_new(theta, E1, E4, C_new, X, multi=True)

    return theta_new    

## --- Calculate Q ---
def calculate_Q(theta_old, X, E1, E2, E4):
    A = theta_old.A
    Gamma = theta_old.Gamma
    C = theta_old.C
    Sigma = theta_old.Sigma
    mu0 = theta_old.mu0
    V0 = theta_old.V0
    N = theta_old.N

    # --- helpers ------------------------------------------------------------
    invV0   = np.linalg.inv(V0)
    invGam  = np.linalg.inv(Gamma)
    invSig  = np.linalg.inv(Sigma)
    logdet  = lambda M: np.linalg.slogdet(M)[1]     # log|M|

    # initial term
    Q_init = -0.5*( logdet(V0)
                    + np.trace(invV0 @ E4[0])
                    - 2*mu0 @ (invV0 @ E1[0])
                    + mu0 @ (invV0 @ mu0) )

    # transition term
    trans_quad = 0.0
    for n in range(1, N):
        trans_quad += np.trace(
           invGam @ (E4[n]
                     - A @ E2[n-1].T
                     - E2[n-1] @ A.T
                     + A @ E4[n-1] @ A.T))
    Q_trans = -0.5*((N-1)*logdet(Gamma) + trans_quad)

    # emission term
    emit_quad = 0.0
    for n,xn in enumerate(X[1:]):
        xn = xn[:,None]               # column vector
        emit_quad += np.trace(
           invSig @ (xn@xn.T
                     - C @ E1[n][:,None] @ xn.T
                     - xn @ E1[n][:,None].T @ C.T
                     + C @ E4[n] @ C.T))
    Q_emit  = -0.5*(N*logdet(Sigma) + emit_quad)

    return Q_init + Q_trans + Q_emit

def calculate_Q_multi(theta, X, E1, E2, E3, E4):
    Nk = theta.Nk
    Q_sum = 0
    for k in range(Nk):
        Q_sum += calculate_Q(theta, X[:,:,k], E1[k], E2[k], E4[k])
    return Q_sum / Nk

def train_EM_single(data, opt):
    max_iter = opt["max_iter"]
    tol = opt["tol"]

    # Initialize parameters
    X = data["X_set"][:,:,0] # [N, Nx]
    theta = initialize_params(data)

    Q_hist = []
    theta_hist = []
    for j in range(max_iter):
        # E-step
        mu, V = kalman_filter(theta, X)
        mu_hat, V_hat = kalman_smoother(theta, X, mu, V)
        # mu_hat, V_hat, mu, V = kalman_filter_filterpy(theta, X)

        # M-step
        E1 = get_E1(mu_hat)
        E2, E3 = get_E2_E3(theta, V, mu_hat, V_hat)
        E4 = get_E4(theta, mu_hat, V_hat)

        theta_new = theta.copy()
        mu0, V0, A, Gamma, C, Sigma = maximization(theta, X, E1, E2, E3, E4)
        theta_new.mu0 = mu0
        theta_new.V0 = V0
        theta_new.A = A
        theta_new.Gamma = Gamma
        theta_new.C = C
        theta_new.Sigma = Sigma

        # Calculate Q
        Q = calculate_Q(theta, X, E1, E2, E4)
        theta = theta_new
        Q_hist.append(Q)
        theta_hist.append(theta.copy())

        print(f"Iteration {j} completed: Q = {Q}")

        # Check convergence
        if len(Q_hist) > 1:
            if np.abs(Q_hist[-1] - Q_hist[-2]) < tol:
                break

    return theta, Q_hist, theta_hist

def train_EM_multi(theta, data, opt):
    max_iter = opt["max_iter"]
    tol = opt["tol"]
    N = theta.N # Number of time steps
    Nk = theta.Nk # Number of trajectories
    Ns = theta.Ns # Number of hidden states 
    Q_hist = []
    theta_hist = []
    
    for j in range(max_iter):

        try:
            # Initialize E1, E2, E3, E4
            E1 = np.empty((0, N, Ns))
            E2 = np.empty((0, N-1, Ns, Ns))
            E3 = np.empty((0, N-1, Ns, Ns))
            E4 = np.empty((0, N, Ns, Ns))

            # Calculate E1, E2, E3, E4 for each trajectory
            for k in range(Nk):
                X_k = data["X_set"][:,:,k]
                mu_k, V_k = kalman_filter(theta, X_k)
                mu_hat_k, V_hat_k = kalman_smoother(theta, X_k, mu_k, V_k)

                E1_k = get_E1(mu_hat_k)
                E2_k, E3_k = get_E2_E3(theta, V_k, mu_hat_k, V_hat_k)
                E4_k = get_E4(theta, mu_hat_k, V_hat_k)

                E1 = np.vstack((E1, [E1_k]))
                E2 = np.vstack((E2, [E2_k]))
                E3 = np.vstack((E3, [E3_k]))
                E4 = np.vstack((E4, [E4_k]))

            X = data["X_set"]
            # Take a summed Maximization step
            theta_new = maximization_multi(theta, X, E1, E2, E3, E4)

            # Calculate Q
            Q = calculate_Q_multi(theta, X, E1, E2, E3, E4)
            if np.isnan(Q):
                print(f"Error occurred at iteration {j}: Q = {Q}")
                break
            theta = theta_new
            Q_hist.append(Q)
            theta_hist.append(theta.copy())

            print(f"Iteration {j} completed: Q = {Q}")

            # Check convergence
            if len(Q_hist) > 1:
                if np.abs(Q_hist[-1] - Q_hist[-2]) < tol:
                    break
        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Returning current results.")
            break

    return theta, Q_hist, theta_hist
    

if __name__ == "__main__":

    # Hyper parameters
    save_name = "simple_linear_model" # Name of the data to load
    # save_name = "spring_damper_perfect"
    opt = {"max_iter": 1000, "tol": 1e-1} # EM algorithm parameters

    try:
        data = np.load(f"asen-5519-projects/data/{save_name}.npz", allow_pickle=True)
    except:
        data = np.load(f"data/{save_name}.npz", allow_pickle=True)

    theta_true = data["theta"].item()
    theta_true.N = data["t_set"].shape[0]
    data_fig, axes = plt.subplots(theta_true.Nx, 1, figsize=(10, 8))
    plot_data(axes, data)

    theta, Q_hist, theta_hist = train_EM_multi(data, opt)
    # theta, Q_hist, theta_hist = train_EM_single(data, opt)

    plot_theta_diffs(theta_hist, theta_true, save_path=f"figs/{save_name}_theta_diffs.png")
    plot_loss(Q_hist, save_path=f"figs/{save_name}_loss.png")

    try:
        vehicle = SimpleLinearModel(theta)
        policy = NoControl()

        t_set = data["t_set"]
        dt = t_set[2,0] - t_set[1,0]
        steps = len(t_set)
        print(f'-- Training EM --\n Max Iter: {opt["max_iter"]}\n Tol: {opt["tol"]}\n steps: {steps}\n dt: {dt}')
        environment = SimpleEnv(steps, dt, vehicle, policy)
        Z, X, t, U = environment.epoch(animate=False, discrete=True)
        pred_data = {"Z_set": np.array(Z), "X_set": np.array(X), "t_set": np.array(t), "U_set": np.array(U)}
        fig = plot_data(axes, data, predicted_data=pred_data, save_path=f"figs/{save_name}_predicted_data.png")
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")

    plt.show()


