import numpy as np
#import matplotlib.pyplot as plt

def trainGP(X, U, length_scale=1.0, sigma_f=1.0, sigma_n=1e-6):
    T = U.shape[0]
    Nx = X.shape[1]
    Nu = U.shape[1]

    Z_train = np.zeros((T, Nx + Nu))
    Y_train = np.zeros((T, Nx))

    for t in range(T):
        Z_train[t] = np.concatenate((X[t], U[t]))
        Y_train[t] = X[t+1] - X[t]

    K = defineKernel(Z_train, Z_train, theta_idx=2, length_scale=length_scale, sigma=sigma_f)
    K += sigma_n**2 * np.eye(T)
    K_inv = np.linalg.inv(K)
    alphas = [K_inv @ Y_train[:, d] for d in range(Nx)]

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