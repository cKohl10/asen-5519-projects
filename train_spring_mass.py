import numpy as np
import matplotlib.pyplot as plt

from environment import SimpleEnv
from vehicle import MassSpringDamper, Theta
from policies import NoControl, SinePolicy, SwitchPolicy
from utils.plotting import plot_eig_stability, plot_data, plot_theta_diffs, plot_loss, print_theta
from utils.EM import train_EM_multi

def spawn_spring_damper(Nx, vel_bound=0.1, noise_level=0.0005, perfect=False):
    # Parameters for the system
    # m1, m2, k1, k2, b, r, cor
    m1 = 5 #Kg
    m2 = 10 #Kg
    k1 = 4 #N/m
    k2 = 5 #N/m
    b = 5 #N*s/m
    r = 0.3 #m
    cor = 0.2 #[-]
    p = np.array([m1, m2, k1, k2, b, r, cor])

    if perfect:
        save_name = "spring_damper_perfect"
        theta_set = {
            "A": np.zeros((6,6)), # Overwriten later
            "Gamma": np.zeros((6,6)), # No state transition noise
            "C": np.eye(6)[:Nx, :], # Output only position
            "Sigma": np.zeros((Nx, Nx)), # No noise on position
            "mu0": np.array([1, 2, 0, 0, 0, 0]).T, # Initial state mean
            "V0": np.array([[0.001, 0, 0, 0, 0, 0],
                            [0, 0.001, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0.001, 0],
                            [0, 0, 0, 0, 0, 0.001]]) * vel_bound, # Initial state covariance on velocity only
            "B": np.zeros((6,2)), # Overwriten later
            "N": steps,
            "Ns": 6,
            "Nx": Nx,
            "Nu": 2,
            "Nk": data_size
        }
    else:
        save_name = "spring_damper_noisy"
        theta_set = {
        "A": np.zeros((6,6)), # Overwriten later
        # "Gamma": np.zeros((6,6)), # No state transition noise
        "Gamma": np.eye(6) * noise_level*0.01, # state transition noise
        "C": np.eye(6)[:Nx, :], # Output only position
        "Sigma": np.eye(Nx) * noise_level, # No noise on position
        "mu0": np.array([1, 2, 0, 0, 0, 0]).T, # Initial state mean
        "V0": np.array([[0.001, 0, 0, 0, 0, 0],
                        [0, 0.001, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0.001, 0],
                        [0, 0, 0, 0, 0, 0.001]]) * vel_bound, # Initial state covariance
        "B": np.zeros((6,2)), # Overwriten later
        "N": steps,
        "Ns": 6,
        "Nx": Nx,
        "Nu": 2,
        "Nk": data_size
        }
    theta = Theta(theta_set)
    vehicle = MassSpringDamper(theta, fig_path=f"figs/{save_name}_dataset")
    vehicle.p_to_dynamics(p)
    theta = vehicle.get_theta()
    plot_eig_stability(theta, f"{save_name} Continuous A", save_path=f"figs/{save_name}_cont_eigs.png")
    return theta, vehicle, save_name

def initialize_params(data):
    # ASSUMES discrete data
    theta = data["theta"].item()

    range = 1e-8 #Initializes the parameters close to the true parameters
    theta.A = theta.A + np.eye(theta.Ns)*range
    theta.Gamma = theta.Gamma + np.eye(theta.Ns)*range
    theta.C = theta.C + np.eye(theta.Ns)[:theta.Nx,:]*range
    theta.Sigma = theta.Sigma + np.eye(theta.Nx)*range
    theta.mu0 = theta.mu0
    theta.V0 = theta.V0

    theta.N = data["t_set"].shape[0]

    # Regularize the covariances parameters
    # theta["Sigma"] = theta["Sigma"] + np.ones((theta["Nx"], theta["Nx"]))*1e-6
    # theta["V0"] = theta["V0"] + np.ones((theta["Ns"], theta["Ns"]))*1e-6
    # theta["Gamma"] = theta["Gamma"] + np.ones((theta["Ns"], theta["Ns"]))*1e-6

    return theta

if __name__ == "__main__":
    # Hyper parameters
    dt = 0.1 # Time step
    steps = 500 # Number of steps to simulate in a sequence
    data_size = 10 # Number of sequences to generate
    vel_bound = 0.5 # Randomized starting velocity bounds
    noise_level = 0.0005 # Gaussian noise level for the position
    perfect = False # Whether to use perfect or noisy dynamics
    Nx = 4 # Number of observable states
    discrete = False # Whether to use discrete or continuous dynamics
    opt = {"max_iter": 1000, "tol": 0.1} # EM algorithm parameters

    policy = NoControl()
    # policy = SinePolicy(w=1, A=1)
    # policy = SwitchPolicy(A=1, t_switch=10)
    theta_true, vehicle, save_name = spawn_spring_damper(Nx, perfect=perfect, vel_bound=vel_bound, noise_level=noise_level) # Spawn the simple linear model

    env = SimpleEnv(steps, dt, vehicle, policy) # Create the environment
    plt.show(block=False)

    env.generate_data(data_size, save_path="data/", save_name=save_name, animate=False, discrete=discrete) # Generate the dataset

    # Load the dataset
    try:
        data = np.load(f"asen-5519-projects/data/{save_name}.npz", allow_pickle=True)
    except:
        data = np.load(f"data/{save_name}.npz", allow_pickle=True)

    theta_true = data["theta"].item()
    theta = initialize_params(data) # Initialize the parameters
    theta, Q_hist, theta_hist = train_EM_multi(theta, data, opt) # Train the EM algorithm
    # theta, Q_hist, theta_hist = train_EM_single(data, opt)

    plot_theta_diffs(theta_hist, theta_true, save_path=f"figs/{save_name}_theta_diffs.png") # Visualize the theta diffs
    plot_loss(Q_hist, save_path=f"figs/{save_name}_loss.png") # Visualize the loss
    print_theta(theta, save_path=f"figs/{save_name}_theta.png", title="Learned")
    print_theta(theta_true, save_path=f"figs/{save_name}_theta_true.png", title="True")

    vehicle = MassSpringDamper(theta) # Simulate a trajectory with the learned parameters

    t_set = data["t_set"]
    print(f'-- Training EM --\n Max Iter: {opt["max_iter"]}\n Tol: {opt["tol"]}\n steps: {steps}\n dt: {dt}')
    environment = SimpleEnv(steps, dt, vehicle, policy)
    Z, X, t, U = environment.epoch(animate=False, discrete=True)
    pred_data = {"Z_set": np.array(Z), "X_set": np.array(X), "t_set": np.array(t), "U_set": np.array(U)}
    fig = plot_data(data, predicted_data=pred_data, save_path=f"figs/{save_name}_predicted_data.png")


    plt.show()
