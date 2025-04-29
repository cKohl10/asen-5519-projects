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
    Z1_rbf = np.delete(Z1, theta_idx, axis=1)
    Z2_rbf = np.delete(Z2, theta_idx, axis=1)
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
