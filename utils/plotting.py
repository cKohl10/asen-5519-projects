import matplotlib.pyplot as plt
import numpy as np
from utils.common import unpack_theta

def plot_data(axes, data, predicted_data=None):
    X_set = data["X_set"]
    t_set = data["t_set"]

    axes[0].plot(t_set[:,0], X_set[:, 0, 0], color='blue', label='Noisy Data')
    if predicted_data is not None:
        pred_x1 = predicted_data["X_set"][:,0]
        t_pred = predicted_data["t_set"]
        axes[0].plot(t_pred, pred_x1, color='red', label='Predicted Data')
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("x1")
    axes[0].set_title("Trajectory x1")
    axes[0].legend()

    axes[1].plot(t_set[:,0], X_set[:, 1, 0], color='blue', label='Noisy Data')
    if predicted_data is not None:
        pred_x2 = predicted_data["X_set"][:,1]
        t_pred = predicted_data["t_set"]
        axes[1].plot(t_pred, pred_x2, color='red', label='Predicted Data')
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("x2")
    axes[1].set_title("Trajectory x2")
    axes[1].legend()

    plt.tight_layout()

def print_theta(theta):
    """Prints the parameters in theta in a formatted way."""
    
    print("--- Theta Parameters ---")
    
    for key, value in theta.items():
        print(f'{key}:')
        if isinstance(value, np.ndarray):
            if value.ndim == 2:
                # Matrix
                print("[")
                for row in value:
                    row_str = "  [" + ", ".join(f"{x: 10.4f}" for x in row) + "]"
                    print(row_str)
                print("]")
            elif value.ndim == 1:
                # Vector
                vec_str = "[" + ", ".join(f"{x: 10.4f}" for x in value) + "]"
                print(vec_str)
            else:
                # Other numpy array (e.g., scalar wrapped in array)
                print(value)
        else:
            # Non-numpy values (like N, Nx, Nu)
            print(value)
        print() # Add a blank line between parameters

def plot_theta_diffs(theta_hist, true_theta):
    num_iterations = len(theta_hist)
    A_diff = []
    Gamma_diff = []
    C_diff = []
    Sigma_diff = []
    mu0_diff = []
    V0_diff = []
    for theta in theta_hist:
        try:
            A, Gamma, C, Sigma, mu0, V0, B, N, Nx, Nu = unpack_theta(theta)
            A_true, Gamma_true, C_true, Sigma_true, mu0_true, V0_true, B_true, N_true, Nx_true, Nu_true = unpack_theta(true_theta)

            A_diff.append(np.linalg.norm(A - A_true))
            Gamma_diff.append(np.linalg.norm(Gamma - Gamma_true))
            C_diff.append(np.linalg.norm(C - C_true))
            Sigma_diff.append(np.linalg.norm(Sigma - Sigma_true))
            mu0_diff.append(np.linalg.norm(mu0 - mu0_true))
            V0_diff.append(np.linalg.norm(V0 - V0_true))
        except:
            A_diff.append(np.nan)
            Gamma_diff.append(np.nan)
            C_diff.append(np.nan)
            Sigma_diff.append(np.nan)
            mu0_diff.append(np.nan)
            V0_diff.append(np.nan)

    def plot_with_nans(ax, data, title):
        data = np.array(data)
        x = np.arange(len(data))
        valid_mask = ~np.isnan(data)
        nan_mask = np.isnan(data)
        
        # Plot valid data in blue
        ax.semilogy(x[valid_mask], data[valid_mask], 'b-', label='Valid')
        # Plot NaN points in red at the bottom of the plot
        if np.any(nan_mask):
            min_val = np.min(data[valid_mask]) if np.any(valid_mask) else 1
            ax.scatter(x[nan_mask], [min_val]*np.sum(nan_mask), color='red', s=10, label='NaN')
        
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Log Difference")
        ax.set_xlim(0, num_iterations - 1)
        if np.any(nan_mask):  # Only show legend if there are NaN values
            ax.legend()

    fig, axes = plt.subplots(2, 3, figsize=(14, 6))
    
    plot_with_nans(axes[0,0], A_diff, "A")
    plot_with_nans(axes[0,1], Gamma_diff, "Gamma")
    plot_with_nans(axes[0,2], C_diff, "C")
    plot_with_nans(axes[1,0], Sigma_diff, "Sigma")
    plot_with_nans(axes[1,1], mu0_diff, "mu0")
    plot_with_nans(axes[1,2], V0_diff, "V0")

    plt.tight_layout()

def plot_loss(Q_hist):
    Q_hist = np.array(Q_hist)
    num_iterations = len(Q_hist)
    x = np.arange(num_iterations)
    
    plt.figure(figsize=(10, 6))
    
    # Split into valid and NaN values
    valid_mask = ~np.isnan(Q_hist)
    nan_mask = np.isnan(Q_hist)
    
    # Plot valid points in blue
    plt.plot(x[valid_mask], Q_hist[valid_mask], 'b-', label='Valid Loss')
    
    # Plot NaN points in red at the bottom of the plot
    if np.any(nan_mask):
        min_val = np.min(Q_hist[valid_mask]) if np.any(valid_mask) else 0
        plt.scatter(x[nan_mask], [min_val]*np.sum(nan_mask), color='red', s=10, label='NaN')
        plt.legend()
    
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss History")
    plt.xlim(0, num_iterations - 1)
    plt.tight_layout()
    # plt.show()