import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Load the data
    data = np.load("data/noisy_data.npz")
    # data = np.load("data/perfect_data.npz")
    Y_set = data["Y_set"]
    t_set = data["t_set"]
    U_set = data["U_set"]

    # Replace the above with the following subplot graph for x1 and x2
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].scatter(t_set[:,0], Y_set[:, 0, 0], s=1)
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("x1")
    axes[0].set_title("Trajectory x1")

    axes[1].scatter(t_set[:,0], Y_set[:, 1, 0], s=1)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("x2")
    axes[1].set_title("Trajectory x2")

    plt.tight_layout()
    plt.show()

