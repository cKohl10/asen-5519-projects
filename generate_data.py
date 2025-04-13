# This script will generate a dataset of states over time for a given policy

import numpy as np
from tqdm import tqdm
from environment import UnboundedPlane
from vehicle import MassSpringDamper
from policies import StepPolicy, NoControl, SinePolicy, StateReferenceFeedbackPolicy
    
def generate_data_no_collision(policy, vehicle, sim_params, save_path=None, save_name=None):
    # Case where the system is localized by a "CV" algorithm with noise and only the position is observed
    # Data will be in the size of (Nx, steps, data_size)
    y_mask = sim_params["y_mask"]
    steps = sim_params["steps"]
    dt = sim_params["dt"]
    data_size = sim_params["data_size"]
    s0_bounds = sim_params["s0_bounds"]
    noise_level = sim_params["noise_level"]
    Nu = vehicle.Nu
    Y_set = np.zeros((steps+1, y_mask.sum(), data_size)) # Noisy data
    t_set = np.zeros((steps+1, data_size))
    U_set = np.zeros((steps, Nu, data_size)) # One less control than steps to account for initial state

    i = 0
    pbar = tqdm(total=data_size, desc="Generating data")
    while i < data_size:
        # We will randomize the starting state
        s0 = np.array([1, 2, 0, 0, 0, 0]) + np.array([np.random.uniform(-s0_bounds[j], s0_bounds[j]) for j in range(len(s0_bounds))])

        vehicle.reset(s0)

        # Load the environment
        environment = UnboundedPlane(steps, dt, vehicle, policy)
        X, t, U, collision_flag = environment.epoch()

        if collision_flag: # Skip if collision occurs
            continue

        # Unwrap list of states into a single array
        X = np.array(X)

        # Add noise to the position
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, (steps+1, y_mask.sum()))
            Y_set[:, :, i] = X[:, y_mask] + noise
        else:
            Y_set[:, :, i] = X[:, y_mask]
        t_set[:, i] = t
        U_set[:, :, i] = U

        pbar.update(1)
        i += 1
    pbar.close()

    if save_path is not None:
        np.savez(save_path + save_name, Y_set=Y_set, t_set=t_set, U_set=U_set)

    return Y_set, t_set, U_set

if __name__ == "__main__":
    # Load the policy
    dt = 0.2
    steps = 1000
    data_size = 100
    vel_bound = 0.2 # Randomized starting velocity bounds
    noise_level = 0.01 # Gaussian noise level for the position

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
    s0 = np.array([0, 0, 0, 0, 0, 0]) # Empty initial state, will be randomised later
    vehicle = MassSpringDamper(s0, p)
    policy = NoControl()

    # Perfect data
    sim_params = {
        "steps": steps,
        "dt": dt,
        "data_size": data_size,
        "noise_level": 0,
        "s0_bounds": np.array([0,0, vel_bound, vel_bound, 0, 0]),
        "y_mask": np.array([True, True, True, True, True, True])
    }
    generate_data_no_collision(policy, vehicle, sim_params, save_path="data/", save_name=f"perfect_data.npz")
    
    # Noisy data
    sim_params = {
        "steps": steps,
        "dt": dt,
        "data_size": data_size,
        "noise_level": noise_level,
        "s0_bounds": np.array([0,0, vel_bound, vel_bound, 0, 0]),
        "y_mask": np.array([True, True, False, False, False, False])
    }
    generate_data_no_collision(policy, vehicle, sim_params, save_path="data/", save_name=f"noisy_data.npz")
