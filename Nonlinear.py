import numpy as np
import matplotlib.pyplot as plt
from GP_Train import defineKernel, trainGP, predictGP, split_data
from tqdm import tqdm

# Load both random and circular motion data for diverse training
print("Loading training data...")
random_data = np.load('data/dubins_random_motion.npz')
circ_data = np.load('data/dubins_circular_motion.npz')

# Combine data from both sources
X_random = random_data['X_set']  # States from random motion
U_random = random_data['U_set']  # Controls from random motion
X_circ = circ_data['X_set'][:,:,0:1]  # First trajectory from circular motion
U_circ = circ_data['U_set'][:,:,0:1]  # First trajectory from circular motion

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
sigma_n = 1e-4  # Increased noise parameter for regularization

print(f"Training on {N_traj} trajectories ({len(traj_indices)} random + 1 circular)")

# Train the Model with adjusted kernel parameters
# Larger length_scale for smoother predictions, higher sigma_f for more flexibility
model = trainGP(X, U, length_scale=2.0, sigma_f=2.0, sigma_n=sigma_n)

# Load in data to test: 
print("\nLoading test data...")
# Use the second trajectory from circular motion for testing
X_test = circ_data['X_set'][:,:,1]  # States
U_test = circ_data['U_set'][:,:,1]  # Controls
t_test = circ_data['t_set'][:,1]    # Time vector

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
plt.subplot(2, 1, 1)
plt.plot(t_test, X_test[:, 0], 'b-', label='True x')
# Start plotting estimates at the end of training data
plt.plot(t_test, estimates[:, 0], 'r--', label='Predicted x')
plt.legend()
plt.title("True vs. Predicted x Position")
plt.xlabel("Time (s)")
plt.ylabel("x Position")
plt.grid(True)

# Plot y position
plt.subplot(2, 1, 2)
plt.plot(t_test, X_test[:, 1], 'b-', label='True y')
# Start plotting estimates at the end of training data
plt.plot(t_test, estimates[:, 1], 'r--', label='Predicted y')
plt.legend()
plt.title("True vs. Predicted y Position")
plt.xlabel("Time (s)")
plt.ylabel("y Position")
plt.grid(True)

plt.tight_layout()
plt.show()