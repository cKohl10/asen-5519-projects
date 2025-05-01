import numpy as np
#import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def trainGP(X, U, length_scale=1.0, sigma_f=1.0, sigma_n=1e-6):
    # Handle both single trajectory and multiple trajectories
    if len(X.shape) == 2:
        # Single trajectory case - convert to 3D array with shape (T, Nx, 1)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        U = U.reshape(U.shape[0], U.shape[1], 1)
    
    T = U.shape[0]
    Nx = X.shape[1]
    Nu = U.shape[1]
    N_traj = X.shape[2]
    
    print(f"Training on {N_traj} trajectories with {T} timesteps each...")
    
    # Initialize arrays to store all trajectories
    Z_train = []
    Y_train = []
    
    # Process each trajectory with progress bar
    for traj_idx in tqdm(range(N_traj), desc="Processing trajectories"):
        X_traj = X[:, :, traj_idx]
        U_traj = U[:, :, traj_idx]
        
        for t in range(T):
            Z_train.append(np.concatenate((X_traj[t], U_traj[t])))
            Y_train.append(X_traj[t+1] - X_traj[t])
    
    # Convert to numpy arrays
    Z_train = np.array(Z_train)
    Y_train = np.array(Y_train)
    
    print(f"Building kernel matrix with {len(Z_train)} training points...")
    
    # Train the model on all trajectories
    K = defineKernel(Z_train, Z_train, theta_idx=2, length_scale=length_scale, sigma=sigma_f)
    K += sigma_n**2 * np.eye(len(Z_train))
    
    print("Computing kernel inverse...")
    K_inv = np.linalg.inv(K)
    
    print("Computing alphas for each dimension...")
    alphas = [K_inv @ Y_train[:, d] for d in range(Nx)]
    
    print("Training complete!")

    return {
        'Z_train': Z_train,
        'Y_train': Y_train,
        'alphas': alphas,
        'kernel_inv': K_inv,
        'params': {'length_scale': length_scale, 'sigma_f': sigma_f, 'sigma_n': sigma_n}
    }

def defineKernel(Z1,Z2,theta_idx=2,length_scale=1.0,sigma=1.0,period = 2*np.pi):
    Z1 = np.atleast_2d(Z1)
    Z2 = np.atleast_2d(Z2)
    
    # Separate out angular dimension (theta)
    theta1 = Z1[:, theta_idx:theta_idx+1]
    theta2 = Z2[:, theta_idx:theta_idx+1]
    dtheta = np.pi * (theta1 - theta2.T) / period
    K_theta = np.exp(-2 * np.sin(dtheta)**2 / length_scale**2)

    # RBF kernel on remaining dimensions
    # Make sure we're handling dimensions consistently
    Z1_rbf = np.delete(Z1, theta_idx, axis=1)
    Z2_rbf = np.delete(Z2, theta_idx, axis=1)
    
    # Ensure dimensions match for dot product
    if Z1_rbf.shape[1] != Z2_rbf.shape[1]:
        raise ValueError(f"Dimension mismatch in RBF kernel computation: {Z1_rbf.shape} vs {Z2_rbf.shape}")
    
    sqdist = np.sum(Z1_rbf**2, axis=1).reshape(-1, 1) + \
            np.sum(Z2_rbf**2, axis=1) - 2 * np.dot(Z1_rbf, Z2_rbf.T)
    K_rbf = np.exp(-0.5 * sqdist / length_scale**2)
    return sigma**2 * K_rbf * K_theta

def predictGP(z_star, model):
    Z_train = model['Z_train']
    alphas = model['alphas']
    params = model['params']

    # Kernel vector between z_star and all training points
    k_star = defineKernel(z_star.reshape(1, -1), Z_train,
                          theta_idx=2,
                          length_scale=params['length_scale'],
                          sigma=params['sigma_f']).flatten()

    # Predict each dimension of Δx
    dx = np.array([k_star @ alpha_d for alpha_d in alphas])

    # Return x_t + Δx as the next state
    return z_star[:3] + dx  # assumes z_star[:3] is [x_t, y_t, θ_t]

def optimize_hyperparameters(X, U, n_trials=10):
    """
    Optimize hyperparameters using a simple random search approach.
    
    Args:
        X: State data
        U: Control data
        n_trials: Number of random trials to perform
        
    Returns:
        Best hyperparameters and their score
    """
    print("Starting hyperparameter optimization...")
    
    # Prepare data
    if len(X.shape) == 2:
        X = X.reshape(X.shape[0], X.shape[1], 1)
        U = U.reshape(U.shape[0], U.shape[1], 1)
    
    T = U.shape[0]
    Nx = X.shape[1]
    Nu = U.shape[1]
    N_traj = X.shape[2]
    
    # Prepare training data
    Z_all = []
    Y_all = []
    
    for traj_idx in range(N_traj):
        X_traj = X[:, :, traj_idx]
        U_traj = U[:, :, traj_idx]
        
        for t in range(T):
            Z_all.append(np.concatenate((X_traj[t], U_traj[t])))
            Y_all.append(X_traj[t+1] - X_traj[t])
    
    Z_all = np.array(Z_all)
    Y_all = np.array(Y_all)
    
    # Split data into training and validation sets
    Z_train, Z_val, Y_train, Y_val = train_test_split(Z_all, Y_all, test_size=0.2, random_state=42)
    
    # Define hyperparameter ranges
    length_scales = np.logspace(-1, 1, 20)
    sigma_fs = np.logspace(-1, 1, 20)
    sigma_ns = np.logspace(-6, -3, 20)
    
    best_score = float('inf')
    best_params = None
    
    # Random search
    for _ in tqdm(range(n_trials), desc="Hyperparameter optimization"):
        # Randomly select hyperparameters
        length_scale = np.random.choice(length_scales)
        sigma_f = np.random.choice(sigma_fs)
        sigma_n = np.random.choice(sigma_ns)
        
        # Build kernel matrix
        K = defineKernel(Z_train, Z_train, theta_idx=2, length_scale=length_scale, sigma=sigma_f)
        K += sigma_n**2 * np.eye(len(Z_train))
        
        try:
            # Compute kernel inverse
            K_inv = np.linalg.inv(K)
            
            # Compute alphas
            alphas = [K_inv @ Y_train[:, d] for d in range(Nx)]
            
            # Compute predictions on validation set
            Y_pred = np.zeros_like(Y_val)
            for i in range(len(Z_val)):
                k_star = defineKernel(Z_val[i:i+1], Z_train, theta_idx=2, 
                                     length_scale=length_scale, sigma=sigma_f).flatten()
                Y_pred[i] = np.array([k_star @ alpha_d for alpha_d in alphas])
            
            # Compute MSE
            mse = np.mean((Y_val - Y_pred)**2)
            
            # Update best parameters if better
            if mse < best_score:
                best_score = mse
                best_params = {
                    'length_scale': length_scale,
                    'sigma_f': sigma_f,
                    'sigma_n': sigma_n
                }
                print(f"New best parameters: {best_params}, MSE: {mse:.6f}")
                
        except np.linalg.LinAlgError:
            # Skip if matrix is singular
            continue
    
    print(f"Optimization complete. Best parameters: {best_params}, MSE: {best_score:.6f}")
    return best_params, best_score

def split_data(data, traj_idx=0, split_index=200):
    # Extract trajectory
    X_full = data['X_set'][:, :, traj_idx]  # shape (T+1, Nx)
    U_full = data['U_set'][:, :, traj_idx]  # shape (T, Nu)

    # Split
    X_train = X_full[:split_index+1]    # for differences: needs +1
    U_train = U_full[:split_index]
    
    X_test = X_full[split_index:]       # will be (T - split_index + 1)
    U_test = U_full[split_index:]       # will be (T - split_index)

    return (X_train, U_train), (X_test, U_test)

def is_positive_definite(matrix):
    """Check if a matrix is positive definite using Cholesky decomposition."""
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False

def check_kernel_properties(K):
    """Check various properties of the kernel matrix."""
    # Check positive definiteness
    is_pd = is_positive_definite(K)
    
    # Compute eigenvalues
    eigvals = np.linalg.eigvals(K)
    min_eigval = np.min(eigvals)
    max_eigval = np.max(eigvals)
    cond_number = max_eigval / min_eigval if min_eigval > 0 else float('inf')
    
    # Check symmetry
    is_symmetric = np.allclose(K, K.T)
    
    return {
        'is_positive_definite': is_pd,
        'min_eigenvalue': min_eigval,
        'max_eigenvalue': max_eigval,
        'condition_number': cond_number,
        'is_symmetric': is_symmetric
    }

def trainGP_V2(X, U, training_points=300, sampling_steps=200, length_scale=1.0, sigma_f=1.0, sigma_n=1e-6):
    """
    Train a GP model on training points and then sequentially sample and update.
    
    Args:
        X: State data of shape (T+1, Nx)
        U: Control data of shape (T, Nu)
        training_points: Number of points to train on initially
        sampling_steps: Number of steps to sample and update
        length_scale: Kernel length scale
        sigma_f: Signal variance
        sigma_n: Noise variance
        
    Returns:
        Dictionary containing the final model and sampled trajectory
    """
    # Ensure inputs are 2D arrays
    X = np.atleast_2d(X)
    U = np.atleast_2d(U)
    
    T = U.shape[0]
    Nx = X.shape[1]
    Nu = U.shape[1]
    
    print(f"Training on {training_points} points...")
    
    # Initialize arrays for training
    Z_train = []
    Y_train = []
    
    # Process training points
    for t in range(min(training_points, T)):
        Z_train.append(np.concatenate((X[t], U[t])))
        Y_train.append(np.concatenate((X[t+1] - X[t], U[t+1] - U[t])))  # Include control changes
    
    # Convert to numpy arrays
    Z_train = np.array(Z_train)
    Y_train = np.array(Y_train)
    
    # Initialize arrays to store sampled trajectory
    sampled_trajectory = []
    sampled_controls = []
    sampled_variances = []  # Store variances at each step
    current_state = X[min(training_points, T)]  # Start from last state of training
    current_control = U[min(training_points, T)]  # Start from last control of training
    sampled_trajectory.append(current_state.copy())
    sampled_controls.append(current_control.copy())
    
    # Main sampling loop
    for step in tqdm(range(sampling_steps), desc="Sampling trajectory"):
        # Build kernel matrix with current training data
        K = defineKernel(Z_train, Z_train, theta_idx=2, length_scale=length_scale, sigma=sigma_f)
        K += sigma_n**2 * np.eye(len(Z_train))
        
        # Add small diagonal term for numerical stability
        K += 1e-10 * np.eye(len(Z_train))
        
        # Compute kernel inverse using Cholesky decomposition for better numerical stability
        try:
            L = np.linalg.cholesky(K)
            K_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(len(K))))
        except np.linalg.LinAlgError:
            # If Cholesky fails, use regular inverse with additional regularization
            K_inv = np.linalg.inv(K + 1e-6 * np.eye(len(K)))
        
        # Compute alphas for each dimension
        alphas = [K_inv @ Y_train[:, d] for d in range(Nx + Nu)]  # Include control dimensions
        
        # Create current state-control pair
        z_star = np.concatenate((current_state, current_control))
        
        # Compute kernel vector between z_star and training points
        k_star = defineKernel(z_star.reshape(1, -1), Z_train,
                             theta_idx=2,
                             length_scale=length_scale,
                             sigma=sigma_f).flatten()
        
        # Compute predictive mean and variance
        mean = np.array([k_star @ alpha_d for alpha_d in alphas])
        
        k_star_star = defineKernel(z_star.reshape(1, -1), z_star.reshape(1, -1),
                                  theta_idx=2,
                                  length_scale=length_scale,
                                  sigma=sigma_f)[0, 0]
        
        # Compute predictive variance for each dimension
        var = np.zeros(Nx + Nu)
        for d in range(Nx + Nu):
            # Compute the quadratic form term
            quadratic_form = k_star @ K_inv @ k_star
            
            # # Check for numerical issues
            # if abs(k_star_star - quadratic_form) < 1e-10:
            #     print(f"\nWarning: Small difference between terms at step {step}, dimension {d}")
            #     print(f"k_star_star: {k_star_star}")
            #     print(f"quadratic_form: {quadratic_form}")
            #     print(f"Difference: {k_star_star - quadratic_form}")
            
            var[d] = k_star_star - quadratic_form
            
            # Ensure variance is non-negative
            var[d] = max(var[d], 1e-10)
        
        # Store variances for this step
        sampled_variances.append(var[:Nx])  # Only store state variances
        
        # Sample from predictive distribution
        sampled_dx = np.random.normal(mean, np.sqrt(var))
        
        # Update state and control
        current_state = current_state + sampled_dx[:Nx]  # First Nx dimensions for state
        current_control = current_control + sampled_dx[Nx:]  # Remaining dimensions for control
        
        # Store sampled state and control
        sampled_trajectory.append(current_state.copy())
        sampled_controls.append(current_control.copy())
        
        # Add new point to training data
        Z_train = np.vstack((Z_train, z_star))
        Y_train = np.vstack((Y_train, sampled_dx))
    
    # Convert sampled trajectory to numpy array
    sampled_trajectory = np.array(sampled_trajectory)  # shape: (sampling_steps+1, Nx)
    sampled_controls = np.array(sampled_controls)  # shape: (sampling_steps+1, Nu)
    sampled_variances = np.array(sampled_variances)  # shape: (sampling_steps, Nx)
    
    # Build final model
    K = defineKernel(Z_train, Z_train, theta_idx=2, length_scale=length_scale, sigma=sigma_f)
    K += sigma_n**2 * np.eye(len(Z_train))
    K += 1e-10 * np.eye(len(Z_train))  # Add small diagonal term
    K_inv = np.linalg.inv(K)
    alphas = [K_inv @ Y_train[:, d] for d in range(Nx + Nu)]
    
    return {
        'Z_train': Z_train,
        'Y_train': Y_train,
        'alphas': alphas,
        'kernel_inv': K_inv,
        'params': {'length_scale': length_scale, 'sigma_f': sigma_f, 'sigma_n': sigma_n},
        'sampled_trajectory': sampled_trajectory,
        'sampled_controls': sampled_controls,
        'sampled_variances': sampled_variances  # Add variances to output
    }
