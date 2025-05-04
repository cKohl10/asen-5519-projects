# This script will generate a dataset of states over time for a given policy

import numpy as np
from tqdm import tqdm
from environment import UnboundedPlane
from vehicle import MassSpringDamper
from policies import StepPolicy, NoControl, SinePolicy, StateReferenceFeedbackPolicy
from utils.plotting import plot_eig_stability
import matplotlib.pyplot as plt

def generate_data(policy, vehicle, theta, sim_params, save_path=None, save_name=None, animate=True):
    # Case where the system is localized by a "CV" algorithm with noise and only the position is observed
    # Data will be in the size of (Nx, steps, data_size)
    steps = sim_params["steps"]
    dt = sim_params["dt"]
    data_size = sim_params["data_size"]
    allow_collision = sim_params["allow_collision"] 
    frame_removal_rate = sim_params["frame_removal_rate"]
    Ns = theta["Ns"]
    Nx = theta["Nx"]
    Nu = theta["Nu"]
    X_set = np.zeros((steps//frame_removal_rate+1, Nx, data_size)) # Noisy data
    t_set = np.zeros((steps//frame_removal_rate+1, data_size))
    U_set = np.zeros((steps//frame_removal_rate, Nu, data_size)) # One less control than steps to account for initial state

    i = 0
    pbar = tqdm(total=data_size, desc="Generating data")
    while i < data_size:
        vehicle.reset()

        environment = UnboundedPlane(steps, dt, vehicle, policy)

        # Load the environment
        if i == 0:
            # X, t, U, collision_flag = environment.epoch(animate=True, save_path=f"animations/mass_spring_damper_{save_name}.gif")
            X, t, U, collision_flag = environment.epoch(animate=animate)
        else:
            X, t, U, collision_flag = environment.epoch()

        if not allow_collision and collision_flag: # Skip if collision occurs
            continue

        # Unwrap list of states into a single array
        X = np.array(X)
        U = np.array(U)
        X_set[:, :, i] = X[::frame_removal_rate, :]
        t_set[:, i] = t[::frame_removal_rate]
        U_set[:, :, i] = U[::frame_removal_rate, :]

        pbar.update(1)
        i += 1

    pbar.close()

    if save_path is not None:
        np.savez(save_path + save_name, X_set=X_set, t_set=t_set, U_set=U_set, theta=theta)

    return X_set, t_set, U_set

if __name__ == "__main__":
    # Load the policy
    dt = 0.1
    steps = 1000
    data_size = 2
    vel_bound = 0.1 # Randomized starting velocity bounds
    noise_level = 0.0005 # Gaussian noise level for the position
    frame_removal_rate = 1 # Take every X frames
    Nx = 6 # Number of observable states

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
    policy = NoControl()

    sim_params = {
        "steps": steps,
        "dt": dt,
        "data_size": data_size,
        "allow_collision": True,
        "frame_removal_rate": frame_removal_rate
    }

    # --- Perfect data ---
    theta = {
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
        "Nu": 2
    }
    vehicle = MassSpringDamper(theta)
    vehicle.p_to_dynamics(p)
    theta = vehicle.get_theta(steps )
    plot_eig_stability(theta, "Perfect Data: A")
    plt.show()
    generate_data(policy, vehicle, theta, sim_params, save_path="data/", save_name=f"perfect_data.npz", animate=True)

    # --- Noisy data ---
    theta = {
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
        "Nu": 2
    }
    vehicle = MassSpringDamper(theta)
    vehicle.p_to_dynamics(p)
    theta = vehicle.get_theta(steps)
    generate_data(policy, vehicle, theta, sim_params, save_path="data/", save_name=f"noisy_data.npz", animate=False)
