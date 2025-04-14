# This script will generate a dataset of states over time for a given policy

import numpy as np
from tqdm import tqdm
from environment import UnboundedPlane
from vehicle import MassSpringDamper
from policies import StepPolicy, NoControl, SinePolicy, StateReferenceFeedbackPolicy
    
def generate_data(policy, vehicle, sim_params, save_path=None, save_name=None):
    # Case where the system is localized by a "CV" algorithm with noise and only the position is observed
    # Data will be in the size of (Nx, steps, data_size)
    steps = sim_params["steps"]
    dt = sim_params["dt"]
    data_size = sim_params["data_size"]
    allow_collision = sim_params["allow_collision"] 
    Nu = vehicle.Nu
    Nx = vehicle.Nx
    X_set = np.zeros((steps+1, Nx, data_size)) # Noisy data
    t_set = np.zeros((steps+1, data_size))
    U_set = np.zeros((steps, Nu, data_size)) # One less control than steps to account for initial state

    i = 0
    pbar = tqdm(total=data_size, desc="Generating data")
    while i < data_size:
        vehicle.reset()

        environment = UnboundedPlane(steps, dt, vehicle, policy)

        # Load the environment
        if i == 0:
            X, t, U, collision_flag = environment.epoch(animate=True)
        else:
            X, t, U, collision_flag = environment.epoch()

        if not allow_collision and collision_flag: # Skip if collision occurs
            continue

        # Unwrap list of states into a single array
        X = np.array(X)
        X_set[:, :, i] = X
        t_set[:, i] = t
        U_set[:, :, i] = U

        pbar.update(1)
        i += 1

    pbar.close()

    if save_path is not None:
        np.savez(save_path + save_name, X_set=X_set, t_set=t_set, U_set=U_set)

    return X_set, t_set, U_set

if __name__ == "__main__":
    # Load the policy
    dt = 0.2
    steps = 1000
    data_size = 2
    vel_bound = 1 # Randomized starting velocity bounds
    noise_level = 0.005 # Gaussian noise level for the position

    # Parameters for the system
    # m1, m2, k1, k2, b, r, cor
    m1 = 10 #Kg
    m2 = 20 #Kg
    k1 = 5 #N/m
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
        "allow_collision": False
    }

    # --- Perfect data ---
    theta = {
        "A": np.zeros((6,6)), # Overwriten later
        "Gamma": np.zeros((6,6)), # No state transition noise
        "C": np.array([[1, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0]]), # Output only position
        "Sigma": np.zeros((2,2)), # No noise on position
        "mu0": np.array([1, 2, 0, 0, 0, 0]), # Initial state mean
        "V0": np.array([[0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0]]) * vel_bound, # Initial state covariance on velocity only
        "B": np.zeros((6,2)) # Overwriten later
    }
    vehicle = MassSpringDamper(theta)
    vehicle.p_to_dynamics(p)
    generate_data(policy, vehicle, sim_params, save_path="data/", save_name=f"perfect_data.npz")

    # --- Noisy data ---
    theta = {
        "A": np.zeros((6,6)), # Overwriten later
        "Gamma": np.zeros((6,6)), # No state transition noise
        # "Gamma": np.array([[1, 0, 0, 0, 0, 0],
        #                    [0, 1, 0, 0, 0, 0],
        #                    [0, 0, 1, 0, 0, 0],
        #                    [0, 0, 0, 1, 0, 0],
        #                    [0, 0, 0, 0, 1, 0],
        #                    [0, 0, 0, 0, 0, 1]]) * noise_level, # state transition noise
        "C": np.array([[1, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0]]), # Output only position
        "Sigma": np.array([[noise_level, 0],
                           [0, noise_level]]), # Gaussian noise on position
        "mu0": np.array([1, 2, 0, 0, 0, 0]), # Initial state mean
        "V0": np.array([[0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0]]) * vel_bound, # Initial state covariance on position only
        "B": np.zeros((6,2)) # Overwriten later
    }
    vehicle = MassSpringDamper(theta)
    vehicle.p_to_dynamics(p)
    generate_data(policy, vehicle, sim_params, save_path="data/", save_name=f"noisy_data.npz")
