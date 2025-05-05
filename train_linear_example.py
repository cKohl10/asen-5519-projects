import numpy as np
import matplotlib.pyplot as plt

from environment import SimpleEnv
from vehicle import SimpleLinearModel, Theta
from policies import NoControl
from utils.plotting import plot_eig_stability, plot_data, plot_theta_diffs, plot_loss
from utils.EM import train_EM_multi

def spawn_simple_example():
    save_name = "simple_linear_model"
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

def initialize_params(data):

    theta = data["theta"].item()
    # Initialize parameters to specified values
    theta.mu0 = np.array([10.0, 10.0, 10.0])
    theta.V0 = np.array([[1.0, 0.5, 0.5],
                             [0.5, 1.0, 0.5],
                             [0.5, 0.5, 1.0]])
    theta.A = np.array([[1.0, 1.1, 1.2],
                             [1.3, 1.4, 1.5],
                             [1.6, 1.7, 1.8]])
    theta.Gamma = np.array([[1.0, 0.5, 0.5],
                                 [0.5, 1.0, 0.5],
                                 [0.5, 0.5, 1.0]])
    theta.C = np.array([[1.0, 1.0, 1.0],
                             [1.0, 1.0, 1.0]])
    theta.Sigma = np.array([[1.0, 0.5],
                                 [0.5, 1.0]])

    return theta

if __name__ == "__main__":
    # Hyper parameters
    dt = 0.1 # Time step
    steps = 500 # Number of steps to simulate in a sequence
    data_size = 1 # Number of sequences to generate
    discrete = True # Whether to use discrete or continuous dynamics
    opt = {"max_iter": 1000, "tol": 10} # EM algorithm parameters

    policy = NoControl()
    theta_true, vehicle, save_name = spawn_simple_example() # Spawn the simple linear model

    env = SimpleEnv(steps, dt, vehicle, policy) # Create the environment
    plt.show(block=False)

    env.generate_data(data_size, save_path="data/", save_name=save_name, animate=False, discrete=discrete) # Generate the dataset

    # Load the dataset
    try:
        data = np.load(f"asen-5519-projects/data/{save_name}.npz", allow_pickle=True)
    except:
        data = np.load(f"data/{save_name}.npz", allow_pickle=True)
    data_fig, axes = plt.subplots(theta_true.Nx, 1, figsize=(10, 8))
    plot_data(axes, data) # Visualize the data being loaded in

    theta = initialize_params(data) # Initialize the parameters
    theta, Q_hist, theta_hist = train_EM_multi(theta, data, opt) # Train the EM algorithm
    # theta, Q_hist, theta_hist = train_EM_single(data, opt)

    plot_theta_diffs(theta_hist, theta_true, save_path=f"figs/{save_name}_theta_diffs.png") # Visualize the theta diffs
    plot_loss(Q_hist, save_path=f"figs/{save_name}_loss.png") # Visualize the loss

    vehicle = SimpleLinearModel(theta) # Simulate a trajectory with the learned parameters

    t_set = data["t_set"]
    print(f'-- Training EM --\n Max Iter: {opt["max_iter"]}\n Tol: {opt["tol"]}\n steps: {steps}\n dt: {dt}')
    environment = SimpleEnv(steps, dt, vehicle, policy)
    Z, X, t, U = environment.epoch(animate=False, discrete=True)
    pred_data = {"Z_set": np.array(Z), "X_set": np.array(X), "t_set": np.array(t), "U_set": np.array(U)}
    fig = plot_data(axes, data, predicted_data=pred_data, save_path=f"figs/{save_name}_predicted_data.png")


    plt.show()
