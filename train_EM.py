import numpy as np
import matplotlib.pyplot as plt

def view_data(data):
    X_set = data["X_set"]
    t_set = data["t_set"]
    U_set = data["U_set"]

    # Replace the above with the following subplot graph for x1 and x2
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].scatter(t_set[:,0], X_set[:, 0, 0], s=1)
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("x1")
    axes[0].set_title("Trajectory x1")

    axes[1].scatter(t_set[:,0], X_set[:, 1, 0], s=1)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("x2")
    axes[1].set_title("Trajectory x2")

    plt.tight_layout()
    plt.show()

def initialize_params(vehicle, data):
    M1 = np.random.uniform(1, 20) # mass 1
    M2 = np.random.uniform(1, 20) # mass 2
    K1 = np.random.uniform(1, 10) # spring constant 1
    K2 = np.random.uniform(1, 10) # spring constant 2
    B = np.random.uniform(1, 10) # damping constant
    r = np.random.uniform(0.1, 0.5) # radius of the masses
    COR = np.random.uniform(0, 1) # Coefficient of restitution [0,1]
    theta = np.array([M1, M2, K1, K2, B, r, COR])

    vehicle.reset()
    vehicle.set_dynamics(theta)

def EM_step(vehicle):
    pass

def train_EM(data, vehicle, opt):

    max_iter = opt["max_iter"]
    tol = opt["tol"]

    # Initialize parameters
    initialize_params(vehicle)

    for i in range(max_iter):
        Q_hist = EM_step()

        error = np.abs(Q_hist[-1] - Q_hist[-2])

        if error < tol:
            break

    return Q_hist

if __name__ == "__main__":
    # Load the data
    data = np.load("data/noisy_data.npz")
    # data = np.load("data/perfect_data.npz")

    view_data(data) # Position data





