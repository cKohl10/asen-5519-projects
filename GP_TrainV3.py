import numpy as np
from tqdm import tqdm

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

def train_gp(Z_train, Y_train_d, Theta_idx,length_scale, sigma_f, sigma_n):
    """Helper function to train a single GP"""
    
    K = defineKernel(Z_train, Z_train, Theta_idx, length_scale=length_scale, sigma=sigma_f)
    K += sigma_n**2 * np.eye(len(Z_train))
    
    try:
        L = np.linalg.cholesky(K)
        K_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(len(K))))
    except np.linalg.LinAlgError:
        K_inv = np.linalg.inv(K)
    
    alpha = K_inv @ Y_train_d
    
    return {
        'alpha': alpha,
        'K': K,
        'kernel_inv': K_inv,
        'params': {'length_scale': length_scale, 'sigma_f': sigma_f, 'sigma_n': sigma_n}
    }
# Train three seperate GPs for each state dimension
def trainGP_V3(X, U, training_points=300, sampling_steps=200, length_scale=1.0, sigma_f=1.0, sigma_n=1e-6, num_trajectories=20,training_method="random"):
    """
    Train three separate GP models (one for each state dimension) on training points and then sequentially sample and update.
    Control is treated as input, and each GP predicts its corresponding state change.
    
    Args:
        X: State data of shape (T+1, Nx)
        U: Control data of shape (T, Nu)
        training_points: Number of points to train on initially
        sampling_steps: Number of steps to sample and update
        length_scale: Kernel length scale
        sigma_f: Signal variance
        sigma_n: Noise variance
        num_trajectories: Number of trajectories to sample
        
    Returns:
        Dictionary containing the three GP models and sampled trajectories
    """
    # Ensure inputs are 2D arrays
    X = np.atleast_2d(X)
    U = np.atleast_2d(U)
    
    T = U.shape[0]
    Nx = X.shape[1]
    Nu = U.shape[1]
    
    print(f"Training GP on {training_points} points...")
    
    # Initialize arrays for training
    initial_Z_train = []  # Store initial training data
    initial_Y_train = []  # Store initial training outputs
    
    # Process training points
    if training_method == "random":
        # Randomly sample points from the trajectory
        random_indices = np.random.choice(T, size=min(training_points, T), replace=False)
        random_indices.sort()  # Sort to maintain temporal order
        
        for t in random_indices:
            # Create input and Output data for each dimension
            initial_Z_train.append(np.concatenate((X[t], U[t])))
            initial_Y_train.append(X[t+1] - X[t]) 
    elif training_method == "sequential":
        for t in range(training_points):
            # Create input and Output data for each dimension
            initial_Z_train.append(np.concatenate((X[t], U[t])))
            initial_Y_train.append(X[t+1] - X[t]) 
    else:
        raise ValueError(f"Invalid training method: {training_method}")

    # Convert to numpy arrays
    initial_Z_train = np.array(initial_Z_train)
    initial_Y_train = np.array(initial_Y_train)
    
    # Initialize arrays to store sampled trajectories
    all_trajectories = []
    all_controls = []
    all_variances = []
    all_state_variances = []
    
    # Generate multiple trajectories
    for traj_idx in tqdm(range(num_trajectories), desc="Generating trajectories"):
        # Reset training data for each trajectory
        Z_train = initial_Z_train.copy()
        Y_train = initial_Y_train.copy()
        
        # Train initial GP
        gp = train_gp(Z_train, Y_train, 2, length_scale, sigma_f, sigma_n)
        
        # Initialize for this trajectory
        sampled_trajectory = []
        sampled_controls = []
        sampled_variances = []
        sampled_state_variance = []
        current_state = X[min(training_points, T)]
        current_control = U[min(training_points, T)]
        sampled_trajectory.append(current_state.copy())
        sampled_controls.append(current_control.copy())
        
        # Main sampling loop for this trajectory
        for step in range(sampling_steps):
            # Create current state-control pair for each dimension
            z_star = np.concatenate((current_state, current_control))
            
            # Initialize arrays for this step's predictions
            mean = np.zeros(Nx)
            var = np.zeros(Nx)
            sampled_dx = np.zeros(Nx)  # Initialize sampled_dx
            
            # Compute kernel vector between z_star and training points
            k_star = defineKernel(z_star.reshape(1, -1), Z_train,
                                    2,
                                    length_scale=gp['params']['length_scale'],
                                    sigma=gp['params']['sigma_f']).flatten()
            
            k_star_star = defineKernel(z_star.reshape(1, -1), z_star.reshape(1, -1),
                                    2,  
                                    length_scale=gp['params']['length_scale'],
                                    sigma=gp['params']['sigma_f'])[0, 0]
            
            # Get predictions from GP for each state
            for d in range(Nx):
                # Compute predictive mean and variance
                mean[d] = k_star @ gp['alpha'].T[d]  # Take d-th column of transposed alpha
                
                # Compute predictive variance for this dimension
                quadratic_form = k_star @ gp['kernel_inv'] @ k_star
                var[d] = k_star_star - quadratic_form
                var[d] = max(var[d], 1e-10)  # Ensure non-negative variance
                
                # Sample from normal distribution
                sampled_dx[d] = np.random.normal(mean[d], np.sqrt(var[d]))
            
            # Update state
            current_state = current_state + sampled_dx
            
            # Add new point to training data
            new_z = np.concatenate((current_state, current_control))
            Z_train = np.concatenate((Z_train, new_z[np.newaxis, :]), axis=0)
            Y_train = np.vstack((Y_train, sampled_dx))
            
            # Retrain GP with new data
            gp = train_gp(Z_train, Y_train, 2, length_scale, sigma_f, sigma_n)
            
            # Store sampled state, control, and variance
            sampled_trajectory.append(current_state.copy())
            sampled_controls.append(current_control.copy())
            sampled_variances.append(var.copy())
            
            # Update state variance
            if step == 0:
                state_var = var.copy()
            else:
                state_var += var
            sampled_state_variance.append(state_var.copy())
        
        # Store this trajectory's data
        all_trajectories.append(np.array(sampled_trajectory))
        all_controls.append(np.array(sampled_controls))
        all_variances.append(np.array(sampled_variances))
        all_state_variances.append(np.array(sampled_state_variance))
    
    return {
        'gp': gp,
        'Z_train': Z_train,
        'Y_train': Y_train,
        'sampled_trajectories': all_trajectories,
        'sampled_controls': all_controls,
        'sampled_variances': all_variances,
        'sampled_state_variances': all_state_variances
    }