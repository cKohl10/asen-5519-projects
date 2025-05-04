# This script will generate a dataset of states over time for a given policy

import numpy as np
from tqdm import tqdm
from environment import SimpleEnv
from vehicle import SimpleLinearModel, Theta, MassSpringDamper
from policies import NoControl
from utils.plotting import plot_eig_stability
import matplotlib.pyplot as plt

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
        save_name = "spring_damper_perfect.npz"
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
        save_name = "spring_damper_noisy.npz"
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
    vehicle = MassSpringDamper(theta, fig_path="figs/spring_damper_dataset")
    vehicle.p_to_dynamics(p)
    theta = vehicle.get_theta()
    plot_eig_stability(theta, "Spring Damper Continuous A", save_path="figs/spring_damper_cont_eigs.png")
    return theta, vehicle, save_name

def spawn_simple_example():
    save_name = "simple_linear_model.npz"
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
    vehicle = SimpleLinearModel(theta, fig_path="figs/simple_linear_model_dataset")
    plot_eig_stability(theta, "Simple Linear Model: A", save_path="figs/simple_linear_model_disc_eigs.png")
    return theta, vehicle, save_name


if __name__ == "__main__":
    # Hyper parameters
    dt = 0.1 # Time step
    steps = 500 # Number of steps to simulate in a sequence
    data_size = 1 # Number of sequences to generate
    vel_bound = 0.1 # Randomized starting velocity bounds
    noise_level = 0.0005 # Gaussian noise level for the position
    perfect = False # Whether to use perfect or noisy dynamics
    Nx = 6 # Number of observable states
    discrete = True # Whether to use discrete or continuous dynamics

    policy = NoControl()
    # theta, vehicle, save_name = spawn_simple_example() # Spawn the simple linear model
    theta, vehicle, save_name = spawn_spring_damper(Nx=Nx, perfect=perfect, vel_bound=vel_bound, noise_level=noise_level) # Spawn the spring damper

    env = SimpleEnv(steps, dt, vehicle, policy)
    plt.show(block=False)
    env.generate_data(data_size, save_path="data/", save_name=save_name, animate=False, discrete=discrete)
