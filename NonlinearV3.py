import numpy as np
import matplotlib.pyplot as plt
from GP_TrainV3 import trainGP_V3
from tqdm import tqdm

def evaluate_model(X, U, length_scale, sigma_f, sigma_n, training_points=200, sampling_steps=300):
    model = trainGP_V3(X, U, 
                    training_points=training_points,
                    sampling_steps=sampling_steps,
                    length_scale=length_scale,
                    sigma_f=sigma_f,
                    sigma_n=sigma_n)
    
    # Calculate RMSE for each dimension
    rmse = np.sqrt(np.mean((X[:sampling_steps+1] - model['sampled_trajectory'])**2, axis=0))
    return rmse, model

print("Loading training data...")
circ_data = np.load('data/dubins_circular_motion.npz')

X = circ_data['X_set'][:,:,0]
U = circ_data['U_set'][:,:,0]
t = circ_data['t_set'][:,0]


model = trainGP_V3(X, U, 
                training_points=200,
                sampling_steps=300,
                length_scale=5.0,
                sigma_f=0.5,
                sigma_n=0.05,
                num_trajectories=5)

# Calculate mean and variance across all sampled trajectories
sampled_trajectories = np.array(model['sampled_trajectories'])  # Shape: (n_trajectories, timesteps, state_dim)
mean_trajectory = np.mean(sampled_trajectories, axis=0)  # Average across trajectories
variance_trajectory = np.var(sampled_trajectories, axis=0)  # Variance across trajectories

print(f"Mean trajectory shape: {mean_trajectory.shape}")
print(f"Variance trajectory shape: {variance_trajectory.shape}")
 

# Plot all sampled trajectories for each state
plt.figure(figsize=(12, 8))

# Plot x position trajectories
plt.subplot(3, 1, 1)
for traj in sampled_trajectories:
    plt.plot(t[200:], traj[:, 0], 'b-', alpha=0.5)
plt.plot(t, X[:, 0], 'r-', linewidth=2, label='True x')
plt.legend()
plt.title(r"True vs. Sampled $x$ Position Trajectories")
plt.xlabel("Time (s)")
plt.ylabel("$x$ Position")
plt.grid(True)

# Plot y position trajectories  
plt.subplot(3, 1, 2)
for traj in sampled_trajectories:
    plt.plot(t[200:], traj[:, 1], 'b-', alpha=0.5)
plt.plot(t, X[:, 1], 'r-', linewidth=2, label='True y')
plt.legend()
plt.title(r"True vs. Sampled $y$ Position Trajectories")
plt.xlabel("Time (s)")
plt.ylabel("$y$ Position")
plt.grid(True)

# Plot theta trajectories
plt.subplot(3, 1, 3)
for traj in sampled_trajectories:
    plt.plot(t[200:], traj[:, 2], 'b-', alpha=0.5)
plt.plot(t, X[:, 2], 'r-', linewidth=2, label=r'True $\theta$')
plt.legend()
plt.title(r"True vs. Sampled $\theta$ Trajectories")
plt.xlabel("Time (s)")
plt.ylabel(r"$\theta$ [rad]")
plt.grid(True)



# Plot the mean and variance of the trajectories
plt.figure(figsize=(12, 8))
# Plot x position
plt.subplot(3, 1, 1)
plt.plot(t, X[:, 0], 'b-', label='True x')
plt.plot(t[200:], mean_trajectory[:, 0], 'r--', label='Mean predicted x')
plt.fill_between(t[200:],
                mean_trajectory[:, 0] - 2*np.sqrt(variance_trajectory[:, 0]),
                mean_trajectory[:, 0] + 2*np.sqrt(variance_trajectory[:, 0]),
                color='g', alpha=0.2, label=r'2$\sigma$ bounds')
plt.legend()
plt.title(r"True vs. Predicted $x$ Position")
plt.xlabel("Time (s)")
plt.ylabel("$x$ Position")
plt.grid(True)

# Plot y position
plt.subplot(3, 1, 2)
plt.plot(t, X[:, 1], 'b-', label='True y')
plt.plot(t[200:], mean_trajectory[:, 1], 'r--', label='Mean predicted y')
plt.fill_between(t[200:],
                mean_trajectory[:, 1] - 2*np.sqrt(variance_trajectory[:, 1]),
                mean_trajectory[:, 1] + 2*np.sqrt(variance_trajectory[:, 1]),
                color='g', alpha=0.2, label=r'2$\sigma$ bounds')
plt.legend()
plt.title(r"True vs. Predicted $y$ Position")
plt.xlabel("Time (s)")
plt.ylabel("$y$ Position")
plt.grid(True)

# Plot theta
plt.subplot(3, 1, 3)
plt.plot(t, X[:, 2], 'b-', label=r'True $\theta$')
plt.plot(t[200:], mean_trajectory[:, 2], 'r--', label=r'Mean predicted $\theta$')
plt.fill_between(t[200:],
                mean_trajectory[:, 2] - 2*np.sqrt(variance_trajectory[:, 2]),
                mean_trajectory[:, 2] + 2*np.sqrt(variance_trajectory[:, 2]),
                color='g', alpha=0.2, label=r'2$\sigma$ bounds')
plt.legend()
plt.title(r"True vs. Predicted $\theta$")
plt.xlabel("Time (s)")
plt.ylabel(r"$\theta$ [rad]")
plt.grid(True)

plt.tight_layout()
plt.show()


