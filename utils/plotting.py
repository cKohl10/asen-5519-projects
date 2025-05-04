import matplotlib.pyplot as plt
import numpy as np
from utils.common import unpack_theta

def plot_data(axes, data, predicted_data=None):
    X_set = data["X_set"]
    t_set = data["t_set"]

    for i in range(X_set.shape[1]):
        axes[i].plot(t_set[:,0], X_set[:, i, 0], color='blue', label='Sensed Data')
        if predicted_data is not None:
            pred_x1 = predicted_data["X_set"][:,i]
            t_pred = predicted_data["t_set"]
            axes[i].plot(t_pred, pred_x1, color='red', label='Predicted Data')
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel(f"x{i+1}")
        axes[i].set_title(f"Trajectory x{i+1}")
        axes[i].legend()

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
            A_diff.append(np.linalg.norm(theta.A - true_theta.A))
            Gamma_diff.append(np.linalg.norm(theta.Gamma - true_theta.Gamma))
            C_diff.append(np.linalg.norm(theta.C - true_theta.C))
            Sigma_diff.append(np.linalg.norm(theta.Sigma - true_theta.Sigma))
            mu0_diff.append(np.linalg.norm(theta.mu0 - true_theta.mu0))
            V0_diff.append(np.linalg.norm(theta.V0 - true_theta.V0))
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
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
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
    Q_hist = np.array(Q_hist[1:])
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

def plot_eig_stability(theta, title):
    A = theta.A
    eig_vals = np.linalg.eigvals(A)
    
    # Plot unit circle
    theta_circle = np.linspace(0, 2*np.pi, 100)
    x_circle = np.cos(theta_circle)
    y_circle = np.sin(theta_circle)

    plt.figure(figsize=(8, 8))
    plt.plot(x_circle, y_circle, 'k--', alpha=0.5, label='Unit Circle')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Plot eigenvalues
    plt.scatter(eig_vals.real, eig_vals.imag, color='blue', label='Eigenvalues')
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.title(title)
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
