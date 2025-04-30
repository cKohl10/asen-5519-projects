import numpy as np
import matplotlib.pyplot as plt
from vehicle import MassSpringDamper
from policies import NoControl
from environment import UnboundedPlane
from scipy.linalg import expm
from utils.plotting import plot_loss, print_theta, plot_theta_diffs, plot_data

# Acknowledgments:
# Tutorial on EM for linear Gaussian systems: https://github.com/tsmatz/hmm-lds-em-algorithm/blob/master/02-lds-em-algorithm.ipynb

def initialize_params(data):
    # M1 = np.random.uniform(1, 20) # mass 1
    # M2 = np.random.uniform(1, 20) # mass 2
    # K1 = np.random.uniform(1, 10) # spring constant 1
    # K2 = np.random.uniform(1, 10) # spring constant 2
    # B = np.random.uniform(1, 10) # damping constant
    # r = np.random.uniform(0.1, 0.5) # radius of the masses
    # COR = np.random.uniform(0, 1) # Coefficient of restitution [0,1]
    # theta = np.array([M1, M2, K1, K2, B, r, COR])

    N = data["X_set"].shape[0]
    Nx = 6
    Nu = data["U_set"].shape[1]
    
    A = np.random.randn(Nx,Nx)
    # Normalize A along its columns
    A = A / np.linalg.norm(A, axis=0)

    Gamma = np.ones((Nx,Nx))*0.01 + np.eye(Nx)*0.01

    C = np.ones((Nu,Nx))

    Sigma = np.array([[1, 0.5], [0.5, 1]])*0.01

    mu0 = np.array([1, 2, 0, 0, 0, 0]).T

    V0 = np.ones((Nx,Nx))*0.01 + np.eye(Nx)*0.01

    B = np.zeros((Nx,Nu))

    theta = {
        "A": A,
        "Gamma": Gamma,
        "C": C,
        "Sigma": Sigma,
        "mu0": mu0,
        "V0": V0,
        "B": B,
        "N": N,
        "Nx": Nx,
        "Nu": Nu
    }

    return theta

def initialize_params_close(data):

    theta = initialize_params_copy(data)

    range = 0.1
    A = theta["A"] + np.random.randn(*theta["A"].shape)*np.linalg.norm(theta["A"])*range
    Gamma = theta["Gamma"] + np.random.randn(*theta["Gamma"].shape)*np.linalg.norm(theta["Gamma"])*range
    C = theta["C"] + np.random.randn(*theta["C"].shape)*np.linalg.norm(theta["C"])*range
    Sigma = theta["Sigma"] + np.random.randn(*theta["Sigma"].shape)*np.linalg.norm(theta["Sigma"])*range
    mu0 = theta["mu0"] + np.random.randn(*theta["mu0"].shape)*np.linalg.norm(theta["mu0"])*range
    V0 = theta["V0"] + np.random.randn(*theta["V0"].shape)*np.linalg.norm(theta["V0"])*range

    theta["A"] = A
    theta["Gamma"] = Gamma
    theta["C"] = C
    theta["Sigma"] = Sigma
    theta["mu0"] = mu0
    theta["V0"] = V0

    theta["N"] = data["X_set"].shape[0]
    theta["Nx"] = 6
    theta["Nu"] = data["X_set"].shape[1]


    return theta

def initialize_params_copy(data):
    true_theta_obj = data["theta"] 
    # true_theta_obj is now a 0-dimensional array containing the dict
    theta = true_theta_obj.item() 
    theta["A"] = expm(theta["A"])
    return theta
    
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

def floor_cov(C, eps=1e-6):
    # symmetrise first
    C = 0.5 * (C + C.T)
    eigval, eigvec = np.linalg.eigh(C)
    eigval = np.clip(eigval, eps, None)
    return (eigvec * eigval) @ eigvec.T

# --- Non-iterative functions ---
def P(V_n, A, Gamma):
    return A @ V_n @ A.T + Gamma

def K(P_nm1, C, Sigma):
    return P_nm1 @ C.T @ np.linalg.pinv(C @ P_nm1 @ C.T + Sigma)

def J(P_n, V_n, A):
    return V_n @ A.T @ np.linalg.pinv(P_n)

# --- Iterative functions ---
def get_V(theta):
    A, Gamma, C, Sigma, _, V0, _, N, Nx, _ = unpack_theta(theta)

    V = np.zeros((N, Nx, Nx))
    K0 = K(V0, C, Sigma)
    # V[0] = V0
    V[0] = (np.eye(Nx) - K0 @ C) @ V0

    for n in range(1,N):
        Pnm1 = P(V[n-1], A, Gamma)
        Kn = K(Pnm1, C, Sigma)
        V[n] = (np.eye(Nx) - Kn @ C) @ Pnm1
    return V

def get_mu(theta, V, X):
    A, Gamma, C, Sigma, mu0, V0, _, N, Nx, _ = unpack_theta(theta)

    mu = np.zeros((N, Nx))
    P0 = P(V[0], A, Gamma)
    K1 = K(P0, C, Sigma)
    # mu[0] = mu0
    mu[0] = mu0 + K1 @ (X[0] - C @ mu0)

    for n in range(1,N):
        Pnm1 = P(V[n-1], A, Gamma)
        Knm1 = K(Pnm1, C, Sigma)
        mu[n] = A @ mu[n-1] + Knm1 @ (X[n] - C @ A @ mu[n-1])

    return mu

def get_mu_hat(theta, mu, V):
    A, Gamma, C, Sigma, mu0, V0, B, N, Nx, Nu = unpack_theta(theta)

    mu_hat_rev = np.empty((0, Nx))
    mu_hat_rev = np.vstack((mu_hat_rev, mu[N-1]))

    for n in range(1, N):
        mu_n = mu[N-n-1]
        mu_hat_nm1 = mu_hat_rev[n-1]
        Vn = V[N-n-1]
        Pn = P(Vn, A, Gamma)
        Jn = J(Pn, Vn, A)
        mu_hat_rev = np.vstack((mu_hat_rev, mu_n + Jn @ (mu_hat_nm1 - A @ mu_n)))

    mu_hat = np.flip(mu_hat_rev, axis=0)

    return mu_hat

def get_V_hat(theta, mu_hat, V):
    A, Gamma, C, Sigma, mu0, V0, B, N, Nx, Nu = unpack_theta(theta)

    V_hat_rev = np.empty((0, Nx, Nx)) # Reverse order of V
    V_hat_rev = np.vstack((V_hat_rev, [V[N-1]]))   

    for n in range(1, N):
        Vn = V[N-n-1]
        Pn = P(Vn, A, Gamma)
        Jn = J(Pn, Vn, A)
        V_hat_rev = np.vstack((V_hat_rev, [Vn + Jn @ (V_hat_rev[n-1] - Pn) @ Jn.T]))

    V_hat = np.flip(V_hat_rev, axis=0)

    return V_hat

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

def get_A_new(theta, E2, E4):
    A, Gamma, C, Sigma, mu0, V0, B, N, Nx, Nu = unpack_theta(theta)
    
    sum_E4 = np.sum(E4[:N-1], axis=0)
    # Add a small regularization term
    regularization = 1e-6 * np.eye(Nx) 
    
    # Check condition number or determinant before inverting (optional debug)
    # print(f"Condition number of sum_E4: {np.linalg.cond(sum_E4)}")
    # print(f"Determinant of sum_E4: {np.linalg.det(sum_E4)}")

    try:
        inv_sum_E4 = np.linalg.inv(sum_E4 + regularization)
        A_new = np.sum(E2, axis=0) @ inv_sum_E4
    except np.linalg.LinAlgError:
        print("Matrix still singular even with regularization!")
        # Handle error appropriately, maybe return old A or use pseudo-inverse
        inv_sum_E4 = np.linalg.pinv(sum_E4) # Use pseudo-inverse as fallback
        A_new = np.sum(E2, axis=0) @ inv_sum_E4


    return A_new

def get_Gamma_new(theta, A_new, E2, E3, E4):
    A, Gamma, C, Sigma, mu0, V0, B, N, Nx, Nu = unpack_theta(theta)

    gamma = np.zeros((Nx,Nx,N-1))

    for n in range(1,N):
        gamma[:,:,n-1] = E4[n] - A_new @ E3[n-1] - E2[n-1] @ A_new.T + A_new @ E4[n-1] @ A_new.T

    Gamma_new = np.sum(gamma, axis=2) / (N-1)

    return Gamma_new

def get_C_new(theta, E1, E4, X):
    A, Gamma, C, Sigma, mu0, V0, B, N, Nx, Nu = unpack_theta(theta)
    # for n in range(N):
    #     elems[:,:,n] = np.outer(X[n], E1[n].T)

    # C_new = np.sum(elems, axis=2) @ np.linalg.inv(np.sum(E4, axis=0))

    def get_C_new_former(E1):
        elems = np.zeros((Nu,Nx,N))
        for n in range(N):
            x_n_T = np.array([X[n]]).T
            E1_n = np.array([E1[n]])
            elems[:,:,n] = x_n_T @ E1_n
        return np.sum(elems, axis=2)
    

    return get_C_new_former(E1) @ np.linalg.inv(np.sum(E4, axis=0))

def get_Sigma_new(theta, E1, E4, C_new, X):
    A, Gamma, C, Sigma, mu0, V0, B, N, Nx, Nu = unpack_theta(theta)

    elems = np.zeros((Nu,Nu,N))
    for n in range(N):
        elems[:,:,n] = X[n] @ X[n].T - C_new @ E1[n] @ X[n].T - np.outer(X[n], E1[n]) @ C_new.T + C_new @ E4[n] @ C_new.T

    Sigma_new = np.sum(elems, axis=2) / N

    return Sigma_new

def train_EM(data, opt):

    max_iter = opt["max_iter"]
    tol = opt["tol"]

    # theta = initialize_params(data)
    theta = initialize_params_close(data)
    # theta = initialize_params_copy(data)
    X = data["X_set"][:,:,0] # [N, Nx]
    theta_hist = []

    Q_hist = [9999]

    for k in range(max_iter):
        
        try:
            # E-step
            Q = 0
            V = get_V(theta)
            mu = get_mu(theta, V, X)
            mu_hat = get_mu_hat(theta, mu, V)
            V_hat = get_V_hat(theta, mu_hat, V)
            # Get E1, E2, E3, E4
            E1 = get_E1(mu_hat)
            E2, E3 = get_E2_E3(theta, V, mu_hat, V_hat)
            E4 = get_E4(theta, mu_hat, V_hat)
            
            # M-step
            mu0_new = get_mu0_new(E1)
            V0_new = get_V0_new(E1, E4)
            A_new = get_A_new(theta, E2, E4)
            Gamma_new = get_Gamma_new(theta, A_new, E2, E3, E4)
            # C_new = get_C_new(theta, E1, E4, X)
            # Sigma_new = get_Sigma_new(theta, E1, E4, C_new, X)
            C_new = theta["C"]
            Sigma_new = theta["Sigma"]

            # Stability is tight for this system, so we need to ensure the eigenvalues of A_new are less than 1
            rho = max(abs(np.linalg.eigvals(A_new)))
            if rho > 1.0:
                A_new /= rho + 1e-3  
            Sigma_new = floor_cov(Sigma_new, eps=1e-6)
            Gamma_new = floor_cov(Gamma_new, eps=1e-6)
            print(f'κ(Sigma)={np.linalg.cond(Sigma_new):.2e}, '
            f'κ(Gamma)={np.linalg.cond(Gamma_new):.2e}, '
            f'κ(C)={np.linalg.cond(C_new):.2e}, '
            f'κ(A)={np.linalg.cond(A_new):.2e}, '
            f'rho(A)={max(abs(np.linalg.eigvals(A_new))):.3f}')

            theta["mu0"] = mu0_new
            theta["V0"] = V0_new
            theta["A"] = A_new
            theta["Gamma"] = Gamma_new
            # theta["C"] = C_new
            # theta["Sigma"] = Sigma_new

            theta_hist.append(theta.copy())
            
            # Q = get_Q(theta, X, E1, E2, E3, E4, V_hat)
            Q_hist.append(Q)

            print(f'Iteration {k}: Q: {Q}')

            error = np.abs(Q_hist[-1] - Q_hist[-2])

            # if error < tol:
            #     break
        except:
            print("Something went wrong with the EM algorithm... Over/underflow?")
            break

    return theta, Q_hist[1:], theta_hist

if __name__ == "__main__":
    # Load the data
    try:
        data = np.load("data/noisy_data.npz", allow_pickle=True)
    except:
        data = np.load("asen-5519-projects/data/noisy_data.npz", allow_pickle=True)

    opt = {
        "max_iter": 100,
        "tol": 1e-6
    }


    theta, Q_hist, theta_hist = train_EM(data, opt)

    plot_loss(Q_hist)
    print_theta(theta)
    true_theta_obj = data["theta"] 
    # true_theta_obj is now a 0-dimensional array containing the dict
    true_theta = true_theta_obj.item() 
    # true_theta is now the actual dictionary
    plot_theta_diffs(theta_hist, true_theta)
    try:
        vehicle = MassSpringDamper(theta)
        policy = NoControl()

        t_set = data["t_set"]
        dt = t_set[2,0] - t_set[1,0]
        steps = len(t_set)
        print(f'-- Training EM --\n Max Iter: {opt["max_iter"]}\n Tol: {opt["tol"]}\n steps: {steps}\n dt: {dt}')
        environment = UnboundedPlane(steps, dt, vehicle, policy, bounds=np.array([100, 100]))
        X, t, U, collision_flag = environment.epoch(animate=False, use_dynamics=False)
        fig = plot_data(data, np.array(X), np.array(t), np.array(U))
    
    except:
        print("Something went wrong with the dynamics")

    plt.show()





