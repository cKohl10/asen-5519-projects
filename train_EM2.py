import numpy as np
import matplotlib.pyplot as plt
from utils.plotting import plot_data, plot_theta_diffs, plot_loss
from utils.common import unpack_theta, reg_inv
from scipy.linalg import expm
from vehicle import MassSpringDamper
from policies import NoControl
from environment import UnboundedPlane


def initialize_params(data):

    dt = data["t_set"][2,0] - data["t_set"][1,0]

    true_theta_obj = data["theta"] 
    # true_theta_obj is now a 0-dimensional array containing the dict
    theta = true_theta_obj.item() 
    theta["A"] = expm(theta["A"]*dt) #<--- Converts the continuous-time system to a discrete-time system

    range = 0.001 #Initializes the parameters close to the true parameters
    theta["A"] = theta["A"] + np.random.randn(*theta["A"].shape)*np.linalg.norm(theta["A"])*range
    theta["Gamma"] = theta["Gamma"] + np.random.randn(*theta["Gamma"].shape)*np.linalg.norm(theta["Gamma"])*range
    theta["C"] = theta["C"] + np.random.randn(*theta["C"].shape)*np.linalg.norm(theta["C"])*range
    theta["Sigma"] = theta["Sigma"] + np.random.randn(*theta["Sigma"].shape)*np.linalg.norm(theta["Sigma"])*range
    theta["mu0"] = theta["mu0"] + np.random.randn(*theta["mu0"].shape)*np.linalg.norm(theta["mu0"])*range
    theta["V0"] = theta["V0"] + np.random.randn(*theta["V0"].shape)*np.linalg.norm(theta["V0"])*range

    # Extra initializations
    theta["N"] = data["X_set"].shape[0]
    theta["Nx"] = 6
    theta["Nu"] = data["X_set"].shape[1]

    return theta
    

## --- Kalman filter ---
def P(V_n, A, Gamma): # Predicted state covariance
    return A @ V_n @ A.T + Gamma

def K(P_nm1, C, Sigma): # Kalman gain
    return P_nm1 @ C.T @ reg_inv(C @ P_nm1 @ C.T + Sigma)

def kalman_filter(theta, X):
    A, Gamma, C, Sigma, mu0, V0, _, N, Nx, Nu = unpack_theta(theta)

    V = np.zeros((N, Nx, Nx))
    mu = np.zeros((N, Nx))
    K1 = K(V0, C, Sigma)

    V[0] = V0
    mu[0] = mu0
    V[1] = (np.eye(Nx) - K1 @ C) @ V0
    mu[1] = mu0 + K1 @ (X[0] - C @ mu0)
    
    for n in range(1, N):
        Pnm1 = P(V[n-1], A, Gamma)
        Kn = K(Pnm1, C, Sigma)

        mu[n] = A @ mu[n-1] + Kn @ (X[n] - C @ A @ mu[n-1])
        V[n] = (np.eye(Nx) - Kn @ C) @ Pnm1

    return mu, V

## --- Kalman smoother ---
def J(P_n, V_n, A):
    return V_n @ A.T @ reg_inv(P_n)

def kalman_smoother(theta, X, mu, V):
    A, Gamma, C, Sigma, mu0, V0, _, N, Nx, Nu = unpack_theta(theta)

    mu_hat_rev = np.empty((0, Nx))
    mu_hat_rev = np.vstack((mu_hat_rev, mu[N-1]))
    V_hat_rev = np.empty((0, Nx, Nx)) # Reverse order of V
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
    A, Gamma, C, Sigma, mu0, V0, B, N, Nx, Nu = unpack_theta(theta)

    E2 = np.zeros((N, Nx, Nx))
    E3 = np.zeros((N, Nx, Nx))

    for n in range(1,N):
        Pnm1 = P(V[n-1], A, Gamma)
        Jnm1 = J(Pnm1, V[n-1], A)
        E2[n] = V_hat[n] @ Jnm1.T + mu_hat[n] @ mu_hat[n-1].T
        E3[n] = E2[n].T

    return E2, E3

def get_E4(theta, mu_hat, V_hat):
    A, Gamma, C, Sigma, mu0, V0, B, N, Nx, Nu = unpack_theta(theta)

    E4 = np.zeros((N, Nx, Nx))

    for n in range(N):
        E4[n] = V_hat[n] + mu_hat[n] @ mu_hat[n].T

    return E4

def get_mu0_new(E1):
    mu0_new = E1[0]
    return mu0_new

def get_V0_new(E1, E4):
    V0_new = E4[0] - E1[0] @ E1[0].T
    return V0_new

def get_A_new(theta, E2, E4, N):    
    sum_E4 = np.sum(E4[:N-1], axis=0)
    A_new = np.sum(E2, axis=0) @ reg_inv(sum_E4)
    return A_new

def get_Gamma_new(theta, A_new, E2, E3, E4, N, Nx):
    gamma = np.zeros((Nx,Nx,N-1))
    for n in range(1,N):
        gamma[:,:,n-1] = E4[n] - A_new @ E3[n-1] - E2[n-1] @ A_new.T + A_new @ E4[n-1] @ A_new.T
    Gamma_new = np.sum(gamma, axis=2) / (N-1)
    return Gamma_new

def get_C_new(theta, E1, E4, X, N, Nu, Nx):
    elems = np.zeros((Nu,Nx,N))
    for n in range(N):
        x_n_T = np.array([X[n]]).T
        E1_n = np.array([E1[n]])
        elems[:,:,n] = x_n_T @ E1_n
    return np.sum(elems, axis=2) @ reg_inv(np.sum(E4, axis=0))

def get_Sigma_new(theta, E1, E4, C_new, X, N, Nu):
    elems = np.zeros((Nu,Nu,N))
    for n in range(N):
        elems[:,:,n] = X[n] @ X[n].T - C_new @ E1[n] @ X[n].T - np.outer(X[n], E1[n]) @ C_new.T + C_new @ E4[n] @ C_new.T
    Sigma_new = np.sum(elems, axis=2) / N
    return Sigma_new

def maximization(theta, X, mu_hat, V_hat, V):
    A, Gamma, C, Sigma, mu0, V0, B, N, Nx, Nu = unpack_theta(theta)
    E1 = get_E1(mu_hat)
    E2, E3 = get_E2_E3(theta, V, mu_hat, V_hat)
    E4 = get_E4(theta, mu_hat, V_hat)

    theta_new = theta.copy()
    theta_new["mu0"] = get_mu0_new(E1)
    theta_new["V0"] = get_V0_new(E1, E4)

    A_new = get_A_new(theta, E2, E4, N)
    theta_new["A"] = A_new
    theta_new["Gamma"] = get_Gamma_new(theta, A_new, E2, E3, E4, N, Nx)

    C_new = get_C_new(theta, E1, E4, X, N, Nu, Nx)
    theta_new["C"] = C_new
    theta_new["Sigma"] = get_Sigma_new(theta, E1, E4, C_new, X, N, Nu)

    return theta_new

## --- Calculate Q ---
def calculate_Q(theta, X, mu_hat, V_hat, V):
    # Simple loss = sum of squared differences between X and C @ mu_hat
    C = theta["C"]
    preds = mu_hat.dot(C.T)              # [N×Nx]·[Nx×Nu] → [N×Nu]
    loss = np.sum((X - preds)**2)        # total squared error
    return loss

def train_EM(data, opt):
    max_iter = opt["max_iter"]
    tol = opt["tol"]

    # Initialize parameters
    X = data["X_set"][:,:,0] # [N, Nx]
    theta = initialize_params(data)

    Q_hist = []
    theta_hist = []
    for k in range(max_iter):
        try:
            # E-step
            mu, V = kalman_filter(theta, X)
            mu_hat, V_hat = kalman_smoother(theta, X, mu, V)

            # M-step
            theta = maximization(theta, X, mu_hat, V_hat, V)

            # Calculate Q
            Q = calculate_Q(theta, X, mu_hat, V_hat, V)
            Q_hist.append(Q)
            theta_hist.append(theta.copy())

            if np.isnan(Q):
                print(f"Error occurred at iteration {k}: Q = {Q}")
                break

            print(f"Iteration {k} completed: Q = {Q}")

            # Check convergence
            # if len(Q_hist) > 1:
            #     if np.abs(Q_hist[-1] - Q_hist[-2]) < tol:
            #         break
        except Exception as e:
            print(f"Error occurred at iteration {k}: {str(e)}")
            break

    return theta, Q_hist, theta_hist

if __name__ == "__main__":
    try:
        data = np.load("asen-5519-projects/data/noisy_data.npz", allow_pickle=True)
    except:
        data = np.load("data/noisy_data.npz", allow_pickle=True)

    data_fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    plot_data(axes, data)

    opt = {"max_iter": 50, "tol": 1e-6}
    theta, Q_hist, theta_hist = train_EM(data, opt)

    plot_theta_diffs(theta_hist, data["theta"].item())
    plot_loss(Q_hist)

    try:
        vehicle = MassSpringDamper(theta)
        policy = NoControl()

        t_set = data["t_set"]
        dt = t_set[2,0] - t_set[1,0]
        steps = len(t_set)
        print(f'-- Training EM --\n Max Iter: {opt["max_iter"]}\n Tol: {opt["tol"]}\n steps: {steps}\n dt: {dt}')
        environment = UnboundedPlane(steps, dt, vehicle, policy, bounds=np.array([100, 100]))
        X, t, U, collision_flag = environment.epoch(animate=False, use_dynamics=False)
        pred_data = {"X_set": np.array(X), "t_set": np.array(t), "U_set": np.array(U)}
        fig = plot_data(axes, data, predicted_data=pred_data)
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")

    plt.show()
