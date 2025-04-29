import numpy as np
import matplotlib.pyplot as plt
from GP_Train import defineKernel, trainGP, predictGP, split_data, optimize_hyperparameters
from tqdm import tqdm

# Load both random and circular motion data for diverse training
print("Loading training data...")
random_data = np.load('data/dubins_random_motion.npz')
circ_data = np.load('data/dubins_circular_motion.npz')

# Combine data from both sources
X_random = random_data['X_set']  # States from random motion
U_random = random_data['U_set']  # Controls from random motion

# Use multiple circular trajectories for training
num_circ_traj = min(10, circ_data['X_set'].shape[2])  # Use up to 3 circular trajectories
X_circ = circ_data['X_set'][:,:,0:num_circ_traj]  # First few trajectories from circular motion
U_circ = circ_data['U_set'][:,:,0:num_circ_traj]  # First few trajectories from circular motion

# Select a subset of random trajectories
traj_indices = np.arange(0, X_random.shape[2], 10)  # Use fewer random trajectories
X_random_subset = X_random[:, :, traj_indices]
U_random_subset = U_random[:, :, traj_indices]

# Combine the data
X = np.concatenate([X_random_subset, X_circ], axis=2)
U = np.concatenate([U_random_subset, U_circ], axis=2)

T = U.shape[0]
Nx = X.shape[1] 
Nu = U.shape[1] 
N_traj = X.shape[2]

print(f"Training on {N_traj} trajectories ({len(traj_indices)} random + {num_circ_traj} circular)")

#Optimize hyperparameters
# print("\nOptimizing hyperparameters...")
# best_params, best_score = optimize_hyperparameters(X, U, n_trials=20)
# print(f"Best hyperparameters: {best_params}")
# print(f"Best validation MSE: {best_score:.6f}")

# # Train the Model with optimized hyperparameters
# model = trainGP(X, U, 
#                 length_scale=best_params['length_scale'], 
#                 sigma_f=best_params['sigma_f'], 
#                 sigma_n=best_params['sigma_n'])

model = trainGP(X, U, 
                length_scale = 10, 
                sigma_f=0.127, 
                sigma_n=7.84*10**-5)

# Load in data to test: 
print("\nLoading test data...")
# Use a different circular trajectory for testing (if available)
test_traj_idx = num_circ_traj  # Use the next trajectory after the ones used for training
if test_traj_idx < circ_data['X_set'].shape[2]:
    X_test = circ_data['X_set'][:,:,test_traj_idx]  # States
    U_test = circ_data['U_set'][:,:,test_traj_idx]  # Controls
    t_test = circ_data['t_set'][:,test_traj_idx]    # Time vector
    print(f"Testing on circular trajectory {test_traj_idx}")
else:
    # Fall back to the first trajectory if we've used all of them
    X_test = circ_data['X_set'][:,:,12]  # States
    U_test = circ_data['U_set'][:,:,12]  # Controls
    t_test = circ_data['t_set'][:,12]    # Time vector
    print("Testing on circular trajectory 0 (all trajectories used for training)")

print(f"Running predictions on test trajectory with {U_test.shape[0]} timesteps...")
x = X_test[0]  # Initial test state
estimates = [x.copy()]

# Add progress bar to prediction loop
for i in tqdm(range(U_test.shape[0]), desc="Making predictions"):
    u = U_test[i]
    z = np.concatenate((x, u))
    x = predictGP(z, model)  # GP returns x_{t+1}
    estimates.append(x)

estimates = np.array(estimates)

# Plot results
plt.figure(figsize=(12, 8))

# Plot x position
plt.subplot(3, 1, 1)
plt.plot(t_test, X_test[:, 0], 'b-', label='True x')
# Start plotting estimates at the end of training data
plt.plot(t_test, estimates[:, 0], 'r--', label='Predicted x')
plt.legend()
plt.title(r"True vs. Predicted $x$ Position")
plt.xlabel("Time (s)")
plt.ylabel("$x$ Position")
plt.grid(True)

# Plot y position
plt.subplot(3, 1, 2)
plt.plot(t_test, X_test[:, 1], 'b-', label='True y')
# Start plotting estimates at the end of training data
plt.plot(t_test, estimates[:, 1], 'r--', label='Predicted y')
plt.legend()
plt.title(r"True vs. Predicted $y$ Position")
plt.xlabel("Time (s)")
plt.ylabel("$y$ Position")
plt.grid(True)

# Plot theta
plt.subplot(3, 1, 3)
plt.plot(t_test, np.mod(X_test[:, 2], 2*np.pi), 'b-', label=r'True $\theta$')
# Start plotting estimates at the end of training data
plt.plot(t_test, np.mod(estimates[:, 2], 2*np.pi), 'r--', label=r'Predicted $\theta$')
plt.legend()
plt.title(r"True vs. Predicted $\theta$")
plt.xlabel("Time (s)")
plt.ylabel(r"$\theta$ [rad]")
plt.grid(True)

plt.tight_layout()

# Plot results
plt.figure(figsize=(12, 8))

# Plot x position
plt.subplot(3, 1, 1)
plt.plot(t_test, X_test[:, 0]-estimates[:, 0], 'b-', label='x Position Error')
plt.legend()
plt.title(r"Error in $x$ Position")
plt.xlabel("Time (s)")
plt.ylabel(r"$x$ Position Error")
plt.grid(True)

# Plot y position
plt.subplot(3, 1, 2)
plt.plot(t_test, X_test[:, 1]-estimates[:, 1], 'b-', label='y Position Error')
plt.legend()
plt.title(r"Error in $y$ Position")
plt.xlabel("Time (s)")
plt.ylabel(r"$y$ Position Error")
plt.grid(True)

# Plot theta
plt.subplot(3, 1, 3)
plt.plot(t_test, X_test[:, 2]-estimates[:, 2], 'b-', label=r'$\theta$ Error')
plt.legend()
plt.title(r"Error in $\theta$ Predictions")
plt.xlabel("Time (s)")
plt.ylabel(r"$\theta$ Error [rad]")
plt.grid(True)

plt.tight_layout()
plt.show()