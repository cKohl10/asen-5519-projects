import matplotlib.pyplot as plt
import numpy as np
from utils.common import unpack_theta

def plot_data(data, predicted_data=None, save_path=None):
    X_set = data["X_set"]
    t_set = data["t_set"]

    fig, axes = plt.subplots(X_set.shape[1], 1, figsize=(10, 8))
    for i in range(X_set.shape[1]):
        # Only add label for sensed data on the first plot
        label = 'Observation Data' if i == 0 else None
        axes[i].plot(t_set[:,0], X_set[:, i, 0], color='blue', label=label)
        if predicted_data is not None:
            pred_x1 = predicted_data["X_set"][:,i]
            t_pred = predicted_data["t_set"]
            # Only add label for predicted data on the first plot
            pred_label = 'Predicted Data' if i == 0 else None
            axes[i].plot(t_pred, pred_x1, color='red', label=pred_label)
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel(f"x{i+1}")
        axes[i].set_title(f"Trajectory x{i+1}")
        if i == 0:
            axes[i].legend()

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)

def print_theta(theta, save_path=None, title=None):
    """Prints the parameters in theta in a formatted way and saves them as tables."""
    
    # Create figure with three subplots stacked vertically
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # First subplot for A and Gamma
    ax1.axis('tight')
    ax1.axis('off')
    
    # Create table for A and Gamma
    A_data = theta.A
    Gamma_data = theta.Gamma
    
    # Combine A and Gamma data
    combined_data = np.vstack((A_data, Gamma_data))
    row_labels = [f'A_{i+1}' for i in range(A_data.shape[0])] + [f'Γ_{i+1}' for i in range(Gamma_data.shape[0])]
    
    # Create table
    table1 = ax1.table(cellText=[[f"{x:.4f}" for x in row] for row in combined_data],
                      rowLabels=row_labels,
                      colLabels=[f'Col {i+1}' for i in range(A_data.shape[1])],
                      cellLoc='center',
                      loc='center')
    
    # Add thick black line above Gamma rows (boundary between A and Gamma)
    boundary_row1 = A_data.shape[0] + 1  # including header row
    for (row, col), cell in table1._cells.items():
        if row == boundary_row1:
            cell.set_edgecolor('black')
            cell.set_linewidth(2)

    table1.auto_set_font_size(False)
    table1.set_fontsize(10)
    table1.scale(1.2, 1.2)
    ax1.set_title(f'{title} State Transition Parameters (A and Γ)')
    
    # Second subplot for C
    ax2.axis('tight')
    ax2.axis('off')
    
    # Create table for C
    C_data = theta.C
    C_row_labels = [f'C_{i+1}' for i in range(C_data.shape[0])]
    C_col_labels = [f'Col {i+1}' for i in range(C_data.shape[1])]
    
    table2 = ax2.table(cellText=[[f"{x:.4f}" for x in row] for row in C_data],
                      rowLabels=C_row_labels,
                      colLabels=C_col_labels,
                      cellLoc='center',
                      loc='center')
    
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1.2, 1.2)
    ax2.set_title(f'{title} Observation Matrix (C)')
    
    # Third subplot for Sigma
    ax3.axis('tight')
    ax3.axis('off')
    
    # Create table for Sigma
    Sigma_data = theta.Sigma
    Sigma_row_labels = [f'Σ_{i+1}' for i in range(Sigma_data.shape[0])]
    Sigma_col_labels = [f'Col {i+1}' for i in range(Sigma_data.shape[1])]
    
    table3 = ax3.table(cellText=[[f"{x:.4f}" for x in row] for row in Sigma_data],
                      rowLabels=Sigma_row_labels,
                      colLabels=Sigma_col_labels,
                      cellLoc='center',
                      loc='center')
    
    table3.auto_set_font_size(False)
    table3.set_fontsize(10)
    table3.scale(1.2, 1.2)
    ax3.set_title(f'{title} Observation Noise Covariance (Σ)')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
    
    # Also print to console
    print("--- Theta Parameters ---")
    print("\nState Transition Parameters:")
    print("A:")
    print(theta.A)
    print("\nGamma:")
    print(theta.Gamma)
    print("\nObservation Parameters:")
    print("C:")
    print(theta.C)
    print("\nSigma:")
    print(theta.Sigma)
    print("\nOther Parameters:")
    print(f"mu0: {theta.mu0}")
    print(f"V0: {theta.V0}")
    print(f"N: {theta.N}")
    print(f"Ns: {theta.Ns}")
    print(f"Nx: {theta.Nx}")
    print(f"Nu: {theta.Nu}")
    print(f"Nk: {theta.Nk}")

def plot_theta_diffs(theta_hist, true_theta, save_path=None):
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
        ax.axhline(y=1, color='r', linestyle='--', alpha=0.5)
        
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
    if save_path is not None:
        plt.savefig(save_path)

def plot_loss(Q_hist, save_path=None):
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
    if save_path is not None:
        plt.savefig(save_path)

def plot_eig_stability(theta, title, save_path=None):
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

    if save_path is not None:
        plt.savefig(save_path)
