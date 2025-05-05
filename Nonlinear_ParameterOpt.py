import numpy as np
from scipy.optimize import minimize
from GP_TrainV3 import trainGP_V3

# Load your data
circ_data = np.load('data/dubins_circular_motion.npz')
X = circ_data['X_set'][:,:,0]
U = circ_data['U_set'][:,:,0]

# Split data into training and validation
T = U.shape[0]
split_idx = int(0.8 * T)
X_train, U_train = X[:split_idx+1], U[:split_idx]
X_val, U_val = X[split_idx:], U[split_idx:]

def rmse_objective(params):
    length_scale, sigma_f, sigma_n = params
    model = trainGP_V3(X_train, U_train, 
                      training_points=min(200, len(U_train)),
                      sampling_steps=len(U_val)-1,
                      length_scale=length_scale,
                      sigma_f=sigma_f,
                      sigma_n=sigma_n,
                      num_trajectories=10)
    mean_trajectory = np.mean(model['sampled_trajectories'], axis=0)
    rmse = np.sqrt(np.mean((X_val[:mean_trajectory.shape[0]] - mean_trajectory)**2))
    print(f"Params: {params}, RMSE: {rmse:.4f}")
    return rmse

initial_params = [5.0, 0.5, 0.05]
bounds = [(0.1, 20.0), (0.01, 2.0), (0.001, 0.1)]

result = minimize(rmse_objective, initial_params, bounds=bounds, method='L-BFGS-B')
print("Optimized parameters:", result.x)
print("Minimum RMSE:", result.fun)
