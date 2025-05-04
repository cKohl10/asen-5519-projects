# This script will generate a dataset of states over time for a given policy

import numpy as np
from tqdm import tqdm
from environment import SimpleEnv
from vehicle import SimpleLinearModel, Theta
from policies import NoControl
from utils.plotting import plot_eig_stability
import matplotlib.pyplot as plt

def generate_data(policy, vehicle, theta, sim_params, save_path=None, save_name=None, animate=True, discrete=True):
    # Case where the system is localized by a "CV" algorithm with noise and only the position is observed
    # Data will be in the size of (Nx, steps, data_size)
    steps = sim_params["steps"]
    dt = sim_params["dt"]
    data_size = sim_params["data_size"]
    frame_removal_rate = sim_params["frame_removal_rate"]
    Ns = theta.Ns
    Nx = theta.Nx
    Nu = theta.Nu
    X_set = np.zeros((steps//frame_removal_rate+1, Nx, data_size)) # Noisy data
    Z_set = np.zeros((steps//frame_removal_rate+1, Ns, data_size)) # Noisy data
    t_set = np.zeros((steps//frame_removal_rate+1, data_size))
    U_set = np.zeros((steps//frame_removal_rate, Nu, data_size)) # One less control than steps to account for initial state

    i = 0
    pbar = tqdm(total=data_size, desc="Generating data")
    while i < data_size:
        vehicle.reset()

        environment = SimpleEnv(steps, dt, vehicle, policy)

        # Load the environment
        if i == 0:
            # X, t, U, collision_flag = environment.epoch(animate=True, save_path=f"animations/mass_spring_damper_{save_name}.gif")
           
            Z, X, t, U = environment.epoch(animate=animate, plot_states=True, discrete=discrete)
        else:
            Z, X, t, U = environment.epoch(discrete=discrete)

        # Unwrap list of states into a single array
        Z = np.array(Z)
        X = np.array(X)
        U = np.array(U)
        Z_set[:, :, i] = Z[::frame_removal_rate, :]
        X_set[:, :, i] = X[::frame_removal_rate, :]
        t_set[:, i] = t[::frame_removal_rate]
        U_set[:, :, i] = U[::frame_removal_rate, :]

        pbar.update(1)
        i += 1

    pbar.close()

    if save_path is not None:
        np.savez(save_path + save_name, Z_set=Z_set, X_set=X_set, t_set=t_set, U_set=U_set, theta=theta)

    return Z_set, X_set, t_set, U_set

if __name__ == "__main__":
    # Load the policy
    dt = 0.1
    steps = 500
    data_size = 10
    vel_bound = 0.1 # Randomized starting velocity bounds
    noise_level = 0.0005 # Gaussian noise level for the position
    frame_removal_rate = 1 # Take every X frames
    policy = NoControl()

    sim_params = {
        "steps": steps,
        "dt": dt,
        "data_size": data_size,
        "frame_removal_rate": frame_removal_rate
    }

    # --- Perfect data ---
    theta_set = {
        "A": np.array([
            [0.750,  0.433, -0.500],
            [-0.217, 0.875,  0.433],
            [0.625, -0.217,  0.750]
        ]),
        "Gamma": np.array([
            [1.5, 0.1, 0.0],
            [0.1, 2.0, 0.3],
            [0.0, 0.3, 1.0]
        ]), 
        "C": np.array([
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0]
        ]),
        "Sigma": np.array([
            [1.0, 0.2],
            [0.2, 2.0]
        ]),
        "mu0": np.array([23, 24, 25]), # Initial state mean
        "V0": np.eye(3)*0.001, # Initial state covariance
        "B": np.zeros((3,2)), # Overwriten later
        "N": steps,
        "Ns": 3,
        "Nx": 2,
        "Nu": 2,
        "Nk": data_size
    }
    theta = Theta(theta_set)
    vehicle = SimpleLinearModel(theta)
    plot_eig_stability(theta, "Simple Linear Model: A")
    plt.show(block=False)
    generate_data(policy, vehicle, theta, sim_params, save_path="data/", save_name=f"simple_linear_model.npz", animate=False, discrete=True)
