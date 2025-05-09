import matplotlib.pyplot as plt
import numpy as np
from utils.common import unpack_theta
import math # Import math for ceil

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

def prediction_plot(data, predicted_data, suptitle=None, save_path=None):
    X_true = data["X_set"]
    Z_true = data["Z_set"]
    t_true = data["t_set"][:,0]
    X_pred = predicted_data["X_set"]
    Z_pred = predicted_data["Z_set"]
    t_pred = predicted_data["t_set"][:,0]

    Nx = X_true.shape[1]
    Ns = Z_true.shape[1]

    axes_labels = ["M_1 Position (m)", 
                   "M_2 Position (m)",
                   "M_1 Velocity (m/s)",
                   "M_2 Velocity (m/s)",
                   "Spring 1 Force (N)",
                   "Spring 2 Force (N)",
                   ]
    
    axes_titles = ["M_1 Position Rollout",
                   "M_2 Position Rollout",
                   "M_1 Velocity Rollout",
                   "M_2 Velocity Rollout",
                   "Spring 1 Force Rollout",
                   "Spring 2 Force Rollout",
                   ]

    # Figure for the observed states X
    if Nx >= 4:
        n_cols_x = 2
        n_rows_x = math.ceil(Nx / n_cols_x)
        fig_x, axes_x = plt.subplots(n_rows_x, n_cols_x, figsize=(10, 2 * n_rows_x))
        axes_x = axes_x.flatten()
    else:
        n_cols_x = 1
        n_rows_x = Nx
        fig_x, axes_x = plt.subplots(n_rows_x, n_cols_x, figsize=(10, 2 * n_rows_x))
        if Nx == 1:
            axes_x = [axes_x]

    for i in range(Nx):
        ax = axes_x[i]
        # Compute y-limits from all true trajectories for this state
        y_true = X_true[:, i, :].flatten()
        y_min = np.min(y_true)
        y_max = np.max(y_true)
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.5 * y_range, y_max + 0.5 * y_range)

        for j in range(X_pred.shape[2]):
            X_pred_j = X_pred[:,:,j]
            if j == 0:
                ax.plot(t_pred, X_pred_j[:,i], color='red', label='Predicted')
            else:
                ax.plot(t_pred, X_pred_j[:,i], color='red', alpha=0.5)
        for j in range(X_true.shape[2]):
            X_true_j = X_true[:,:,j]
            if j == 0:
                ax.plot(t_true, X_true_j[:,i], color='blue', label='True')
            else:
                ax.plot(t_true, X_true_j[:,i], color='blue', alpha=0.2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(axes_labels[i])
        ax.set_title("Observed " + axes_titles[i])
        ax.grid(True)
        if i == 0:
            ax.legend()
    fig_x.suptitle(suptitle + " Observed States")
    plt.tight_layout()

    # Figure for the latent states Z
    if Ns >= 6:
        n_cols_z = 2
        n_rows_z = math.ceil(Ns / n_cols_z)
        fig_z, axes_z = plt.subplots(n_rows_z, n_cols_z, figsize=(10, 2 * n_rows_z))
        axes_z = axes_z.flatten()
    else:
        n_cols_z = 1
        n_rows_z = Ns
        fig_z, axes_z = plt.subplots(n_rows_z, n_cols_z, figsize=(10, 2 * n_rows_z))
        if Ns == 1:
            axes_z = [axes_z]

    for i in range(Ns):
        ax = axes_z[i]
        # Compute y-limits from all true trajectories for this latent state
        y_true = Z_true[:, i, :].flatten()
        y_min = np.min(y_true)
        y_max = np.max(y_true)
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.5 * y_range, y_max + 0.5 * y_range)

        for j in range(Z_pred.shape[2]):
            Z_pred_j = Z_pred[:,:,j]
            if j == 0:
                ax.plot(t_pred, Z_pred_j[:,i], color='red', label='Predicted')
            else:
                ax.plot(t_pred, Z_pred_j[:,i], color='red', alpha=0.5)
        for j in range(Z_true.shape[2]):
            Z_true_j = Z_true[:,:,j]
            if j == 0:
                ax.plot(t_true, Z_true_j[:,i], color='blue', label='True')
            else:
                ax.plot(t_true, Z_true_j[:,i], color='blue', alpha=0.2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(axes_labels[i])
        ax.set_title("Latent " + axes_titles[i])
        ax.grid(True)
        if i == 0:
            ax.legend()
    fig_z.suptitle(suptitle + " Latent States")
    plt.tight_layout()

    if save_path is not None:
        fig_x.savefig(save_path+"_x.png")
        fig_z.savefig(save_path+"_z.png")

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
        
def plot_eig_stability_compare(theta_true, theta_learned, title, save_path=None):
    A_true = theta_true.A
    A_learned = theta_learned.A
    eig_vals_true = np.linalg.eigvals(A_true)
    eig_vals_learned = np.linalg.eigvals(A_learned)
        
    # Plot unit circle
    theta_circle = np.linspace(0, 2*np.pi, 100)
    x_circle = np.cos(theta_circle)
    y_circle = np.sin(theta_circle)

    plt.figure(figsize=(8, 8))
    plt.plot(x_circle, y_circle, 'k--', alpha=0.5, label='Unit Circle')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Plot eigenvalues
    plt.scatter(eig_vals_true.real, eig_vals_true.imag, color='blue', label='True Dynamics')
    plt.scatter(eig_vals_learned.real, eig_vals_learned.imag, color='red', label='Learned Dynamics')
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.title(title)
    plt.legend()
    plt.axis('equal')
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path)

def plot_eig_stability_compare_hist(theta_true, theta_learned_hist, title, save_path=None):
    A_true = theta_true.A
    eig_vals_true = np.linalg.eigvals(A_true)
        
    # Plot unit circle
    theta_circle = np.linspace(0, 2*np.pi, 100)
    x_circle = np.cos(theta_circle)
    y_circle = np.sin(theta_circle)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x_circle, y_circle, 'k--', alpha=0.5, label='Unit Circle')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Limit to at most 200 theta values (excluding the final one which we plot separately)
    max_thetas = 200
    num_iterations = len(theta_learned_hist)
    
    if num_iterations > max_thetas + 1:  # +1 for the final theta we plot separately
        # Create evenly spaced indices to sample from theta_learned_hist
        # Exclude the last element which will be plotted separately
        indices = np.linspace(0, num_iterations-2, max_thetas, dtype=int)
        sampled_thetas = [theta_learned_hist[i] for i in indices]
    else:
        sampled_thetas = theta_learned_hist[:-1]
    
    # Create a colormap from dark red to light red
    colors = plt.cm.Reds(np.linspace(0.3, 1.0, len(sampled_thetas)))
    
    # Plot eigenvalues with a color gradient (excluding the final theta)
    for i, theta in enumerate(sampled_thetas):
        A_learned = theta.A
        eig_vals_learned = np.linalg.eigvals(A_learned)
        ax.scatter(eig_vals_learned.real, eig_vals_learned.imag, 
                  color=colors[i], 
                  label='Learned Dynamics' if i == 0 else None)
    
    # Plot the final theta eigenvalues as stars
    if len(theta_learned_hist) > 0:
        A_final = theta_learned_hist[-1].A
        eig_vals_final = np.linalg.eigvals(A_final)
        ax.scatter(eig_vals_final.real, eig_vals_final.imag,
                  color='pink', marker='*', s=200, label='Final Iteration',
                  edgecolor='black', linewidth=0.5, zorder=10)
    
    # Plot true eigenvalues last so they're on top
    ax.scatter(eig_vals_true.real, eig_vals_true.imag, 
              color='blue', label='True Dynamics',
              zorder=11)  # Higher zorder to appear on top
    
    # Add colorbar
    norm = plt.Normalize(0, num_iterations-2)  # Use the original iteration count for the colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Iteration')
    
    # Set custom ticks on colorbar for start, middle, and end iterations
    if num_iterations > 2:
        cbar.set_ticks([0, (num_iterations-2)/2, num_iterations-2])
        cbar.set_ticklabels([0, num_iterations//2, num_iterations-2])
    
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.set_title(title)
    ax.legend()
    ax.axis('equal')
    ax.grid(True)

    if save_path is not None:
        plt.savefig(save_path)
