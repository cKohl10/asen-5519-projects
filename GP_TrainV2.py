import numpy as np
#import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

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
    
    # Ensure dimensions match for dot product
    if Z1_rbf.shape[1] != Z2_rbf.shape[1]:
        raise ValueError(f"Dimension mismatch in RBF kernel computation: {Z1_rbf.shape} vs {Z2_rbf.shape}")
    
    sqdist = np.sum(Z1_rbf**2, axis=1).reshape(-1, 1) + \
            np.sum(Z2_rbf**2, axis=1) - 2 * np.dot(Z1_rbf, Z2_rbf.T)
    K_rbf = np.exp(-0.5 * sqdist / (length_scale**2))
    
    # Combine kernels with proper scaling
    return sigma**2 * K_rbf * K_theta


def trainGP_V2(X, U, training_points=300, sampling_steps=200, length_scale=1.0, sigma_f=1.0, sigma_n=1e-6):
    """
    Train a GP model on training points and then sequentially sample and update.
    Control is treated as input, and only state changes are predicted.
    
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
    Z_train = []  # Input: [state, control]
    Y_train = []  # Output: state changes only
    
    # Process training points
    for t in range(min(training_points, T)):
        Z_train.append(np.concatenate((X[t], U[t])))  # Input: current state and control
        Y_train.append(X[t+1] - X[t])  # Output: only state changes
    
    # Convert to numpy arrays
    Z_train = np.array(Z_train)
    Y_train = np.array(Y_train)
    
    # Initialize arrays to store sampled trajectory
    sampled_trajectory = []
    sampled_controls = []
    sampled_variances = []
    current_state = X[min(training_points, T)]
    current_control = U[min(training_points, T)]
    sampled_trajectory.append(current_state.copy())
    sampled_controls.append(current_control.copy())
    
    # Main sampling loop
    for step in tqdm(range(sampling_steps), desc="Sampling trajectory"):
        # Create current state-control pair
        z_star = np.concatenate((current_state, current_control))
        
        # Build kernel matrix with current training data
        K = defineKernel(Z_train, Z_train, theta_idx=2, length_scale=length_scale, sigma=sigma_f)
        K += sigma_n**2 * np.eye(len(Z_train))
        K += 1e-6 * np.eye(len(Z_train))
        
        # Compute kernel inverse using Cholesky decomposition
        try:
            L = np.linalg.cholesky(K)
            K_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(len(K))))
        except np.linalg.LinAlgError:
            K_inv = np.linalg.inv(K + 1e-6 * np.eye(len(K)))
        
        # Compute alphas for each dimension (only state dimensions)
        alphas = [K_inv @ Y_train[:, d] for d in range(Nx)]
        
        # Compute kernel vector between z_star and training points
        k_star = defineKernel(z_star.reshape(1, -1), Z_train,
                             theta_idx=2,
                             length_scale=length_scale,
                             sigma=sigma_f).flatten()
        
        # Compute predictive mean and variance (only for state changes)
        mean = np.array([k_star @ alpha_d for alpha_d in alphas])
        
        k_star_star = defineKernel(z_star.reshape(1, -1), z_star.reshape(1, -1),
                                  theta_idx=2,
                                  length_scale=length_scale,
                                  sigma=sigma_f)[0, 0]
        
        # Compute predictive variance for each state dimension
        var = np.zeros(Nx)
        for d in range(Nx):
            # Compute the quadratic form term using the alpha vector for this dimension
            quadratic_form = k_star @ K_inv @ k_star
            # Add the contribution from the alpha vector for this dimension
            var[d] = k_star_star - quadratic_form + sigma_n**2
            var[d] = max(var[d], 1e-10)  # Ensure non-negative variance
            

        # Store variances for this step
        sampled_variances.append(var)  # Store all state variances
        
        # Sample from predictive distribution (only state changes)
        sampled_dx = np.random.normal(mean, np.sqrt(var))
        
        # Update state (control remains unchanged)
        current_state = current_state + sampled_dx
        
        # Add new point to training data
        Z_train = np.vstack((Z_train, z_star))
        Y_train = np.vstack((Y_train, sampled_dx))
        
        # Store sampled state and control
        sampled_trajectory.append(current_state.copy())
        sampled_controls.append(current_control.copy())
    
    # Convert sampled trajectory to numpy array
    sampled_trajectory = np.array(sampled_trajectory)  # shape: (sampling_steps+1, Nx)
    sampled_controls = np.array(sampled_controls)  # shape: (sampling_steps+1, Nu)
    sampled_variances = np.array(sampled_variances)  # shape: (sampling_steps, Nx)
    
    return {
        'Z_train': Z_train,
        'Y_train': Y_train,
        'alphas': alphas,
        'kernel_inv': K_inv,
        'params': {'length_scale': length_scale, 'sigma_f': sigma_f, 'sigma_n': sigma_n},
        'sampled_trajectory': sampled_trajectory,
        'sampled_controls': sampled_controls,
        'sampled_variances': sampled_variances
    }