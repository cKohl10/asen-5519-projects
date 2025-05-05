import numpy as np
import matplotlib.pyplot as plt
from GP_TrainV3 import trainGP_V3
from tqdm import tqdm

# Set default font sizes for LaTeX compatibility
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.figsize': (10, 8)  # Smaller figure size
})

print("Loading training data...")
circ_data = np.load('data/dubins_circular_motion.npz')

X = circ_data['X_set'][:,:,0]
U = circ_data['U_set'][:,:,0]
t = circ_data['t_set'][:,0]

num_train = 300;
num_test = 500-num_train;

model_random = trainGP_V3(X, U, 
                training_points=num_train,
                sampling_steps=num_test,
                length_scale=5,
                sigma_f=0.5,
                sigma_n=0.05,
                num_trajectories=30,training_method="random")

model_sequential = trainGP_V3(X, U, 
                training_points=num_train,
                sampling_steps=num_test,
                length_scale=5,
                sigma_f=0.5,
                sigma_n=0.05,
                num_trajectories=30,training_method="sequential")

# Calculate mean and variance across all sampled trajectories
sampled_trajectories_random = np.array(model_random['sampled_trajectories'])  # Shape: (n_trajectories, timesteps, state_dim)
mean_trajectory_random = np.mean(sampled_trajectories_random, axis=0)  # Average across trajectories
variance_trajectory_random = np.var(sampled_trajectories_random, axis=0)  # Variance across trajectories

sampled_trajectories_sequential = np.array(model_sequential['sampled_trajectories'])  # Shape: (n_trajectories, timesteps, state_dim)
mean_trajectory_sequential = np.mean(sampled_trajectories_sequential, axis=0)  # Average across trajectories
variance_trajectory_sequential = np.var(sampled_trajectories_sequential, axis=0)  # Variance across trajectories
 

# Plot all sampled trajectories for each state
plt.figure(figsize=(10, 8))

# Plot x position trajectories
plt.subplot(3, 1, 1)
for traj in sampled_trajectories_random:
    plt.plot(t[num_train:], traj[:, 0], 'b-', alpha=0.5)
plt.plot(t[num_train:], X[num_train:, 0], 'r-', linewidth=2, label='True x')
plt.legend(fontsize=14)
plt.title(r"True vs. Sampled $x$ Position Trajectories", fontsize=18)
plt.xlabel("Time (s)", fontsize=16)
plt.ylabel("$x$ Position", fontsize=16)
plt.grid(True)

# Plot y position trajectories  
plt.subplot(3, 1, 2)
for traj in sampled_trajectories_random:
    plt.plot(t[num_train:], traj[:, 1], 'b-', alpha=0.5)
plt.plot(t[num_train:], X[num_train:, 1], 'r-', linewidth=2, label='True y')
plt.legend(fontsize=14)
plt.title(r"True vs. Sampled $y$ Position Trajectories", fontsize=18)
plt.xlabel("Time (s)", fontsize=16)
plt.ylabel("$y$ Position", fontsize=16)
plt.grid(True)

# Plot theta trajectories
plt.subplot(3, 1, 3)
for traj in sampled_trajectories_random:
    plt.plot(t[num_train:], traj[:, 2], 'b-', alpha=0.5)
plt.plot(t[num_train:], X[num_train:, 2], 'r-', linewidth=2, label=r'True $\theta$')
plt.legend(fontsize=14)
plt.title(r"True vs. Sampled $\theta$ Trajectories", fontsize=18)
plt.xlabel("Time (s)", fontsize=16)
plt.ylabel(r"$\theta$ [rad]", fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.suptitle(r"Random Training Sampled Trajectories", fontsize=20)
plt.subplots_adjust(hspace=0.54,top=0.9)  # Increased spacing between subplots

# Plot all sampled trajectories for each state
plt.figure(figsize=(10, 8))

# Plot x position trajectories
plt.subplot(3, 1, 1)
for traj in sampled_trajectories_sequential:
    plt.plot(t[num_train:], traj[:, 0], 'b-', alpha=0.5)
plt.plot(t[num_train:], X[num_train:, 0], 'r-', linewidth=2, label='True x')
plt.legend(fontsize=14)
plt.title(r"True vs. Sampled $x$ Position Trajectories", fontsize=18)
plt.xlabel("Time (s)", fontsize=16)
plt.ylabel("$x$ Position", fontsize=16)
plt.grid(True)

# Plot y position trajectories  
plt.subplot(3, 1, 2)
for traj in sampled_trajectories_sequential:
    plt.plot(t[num_train:], traj[:, 1], 'b-', alpha=0.5)
plt.plot(t[num_train:], X[num_train:, 1], 'r-', linewidth=2, label='True y')
plt.legend(fontsize=14)
plt.title(r"True vs. Sampled $y$ Position Trajectories", fontsize=18)
plt.xlabel("Time (s)", fontsize=16)
plt.ylabel("$y$ Position", fontsize=16)
plt.grid(True)

# Plot theta trajectories
plt.subplot(3, 1, 3)
for traj in sampled_trajectories_sequential:
    plt.plot(t[num_train:], traj[:, 2], 'b-', alpha=0.5)
plt.plot(t[num_train:], X[num_train:, 2], 'r-', linewidth=2, label=r'True $\theta$')
plt.legend(fontsize=14)
plt.title(r"True vs. Sampled $\theta$ Trajectories", fontsize=18)
plt.xlabel("Time (s)", fontsize=16)
plt.ylabel(r"$\theta$ [rad]", fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.subplots_adjust(hspace=0.54,top=0.9)  # Increased spacing between subplots
plt.suptitle(r"Sequential Training Sampled Trajectories", fontsize=20)

# Plot the mean and variance of the trajectories
plt.figure(figsize=(10, 8))
# Plot x position
plt.subplot(3, 1, 1)
plt.plot(t[num_train:], X[num_train:, 0], 'b-', label='True x')
plt.plot(t[num_train:], mean_trajectory_random[:, 0], 'r--', label='Mean predicted x')
plt.fill_between(t[num_train:],
                mean_trajectory_random[:, 0] - 2*np.sqrt(variance_trajectory_random[:, 0]),
                mean_trajectory_random[:, 0] + 2*np.sqrt(variance_trajectory_random[:, 0]),
                color='g', alpha=0.2, label=r'2$\sigma$ bounds')
plt.legend(fontsize=14)
plt.title(r"True vs. Predicted $x$ Position", fontsize=18)
plt.xlabel("Time (s)", fontsize=16)
plt.ylabel("$x$ Position", fontsize=16)
plt.grid(True)

# Plot y position
plt.subplot(3, 1, 2)
plt.plot(t[num_train:], X[num_train:, 1], 'b-', label='True y')
plt.plot(t[num_train:], mean_trajectory_random[:, 1], 'r--', label='Mean predicted y')
plt.fill_between(t[num_train:],
                mean_trajectory_random[:, 1] - 2*np.sqrt(variance_trajectory_random[:, 1]),
                mean_trajectory_random[:, 1] + 2*np.sqrt(variance_trajectory_random[:, 1]),
                color='g', alpha=0.2, label=r'2$\sigma$ bounds')
plt.legend(fontsize=14)
plt.title(r"True vs. Predicted $y$ Position", fontsize=18)
plt.xlabel("Time (s)", fontsize=16)
plt.ylabel("$y$ Position", fontsize=16)
plt.grid(True)

# Plot theta
plt.subplot(3, 1, 3)
plt.plot(t[num_train:], X[num_train:, 2], 'b-', label=r'True $\theta$')
plt.plot(t[num_train:], mean_trajectory_random[:, 2], 'r--', label=r'Mean predicted $\theta$')
plt.fill_between(t[num_train:],
                mean_trajectory_random[:, 2] - 2*np.sqrt(variance_trajectory_random[:, 2]),
                mean_trajectory_random[:, 2] + 2*np.sqrt(variance_trajectory_random[:, 2]),
                color='g', alpha=0.2, label=r'2$\sigma$ bounds')
plt.legend(fontsize=14)
plt.title(r"True vs. Predicted $\theta$", fontsize=18)
plt.xlabel("Time (s)", fontsize=16)
plt.ylabel(r"$\theta$ [rad]", fontsize=16)
plt.grid(True)
plt.suptitle(r"Random Training Predicted Trajectory", fontsize=20)
plt.tight_layout()
plt.subplots_adjust(hspace=0.54,top=0.9)  # Increased spacing between subplots



# Plot the mean and variance of the trajectories
plt.figure(figsize=(10, 8))
# Plot x position
plt.subplot(3, 1, 1)
plt.plot(t[num_train:], X[num_train:, 0], 'b-', label='True x')
plt.plot(t[num_train:], mean_trajectory_sequential[:, 0], 'r--', label='Mean predicted x')
plt.fill_between(t[num_train:],
                mean_trajectory_sequential[:, 0] - 2*np.sqrt(variance_trajectory_sequential[:, 0]),
                mean_trajectory_sequential[:, 0] + 2*np.sqrt(variance_trajectory_sequential[:, 0]),
                color='g', alpha=0.2, label=r'2$\sigma$ bounds')
plt.legend(fontsize=14)
plt.title(r"True vs. Predicted $x$ Position", fontsize=18)
plt.xlabel("Time (s)", fontsize=16)
plt.ylabel("$x$ Position", fontsize=16)
plt.grid(True)

# Plot y position
plt.subplot(3, 1, 2)
plt.plot(t[num_train:], X[num_train:, 1], 'b-', label='True y')
plt.plot(t[num_train:], mean_trajectory_sequential[:, 1], 'r--', label='Mean predicted y')
plt.fill_between(t[num_train:],
                mean_trajectory_sequential[:, 1] - 2*np.sqrt(variance_trajectory_sequential[:, 1]),
                mean_trajectory_sequential[:, 1] + 2*np.sqrt(variance_trajectory_sequential[:, 1]),
                color='g', alpha=0.2, label=r'2$\sigma$ bounds')
plt.legend(fontsize=14)
plt.title(r"True vs. Predicted $y$ Position", fontsize=18)
plt.xlabel("Time (s)", fontsize=16)
plt.ylabel("$y$ Position", fontsize=16)
plt.grid(True)

# Plot theta
plt.subplot(3, 1, 3)
plt.plot(t[num_train:], X[num_train:, 2], 'b-', label=r'True $\theta$')
plt.plot(t[num_train:], mean_trajectory_sequential[:, 2], 'r--', label=r'Mean predicted $\theta$')
plt.fill_between(t[num_train:],
                mean_trajectory_sequential[:, 2] - 2*np.sqrt(variance_trajectory_sequential[:, 2]),
                mean_trajectory_sequential[:, 2] + 2*np.sqrt(variance_trajectory_sequential[:, 2]),
                color='g', alpha=0.2, label=r'2$\sigma$ bounds')
plt.legend(fontsize=14)
plt.title(r"True vs. Predicted $\theta$", fontsize=18)
plt.xlabel("Time (s)", fontsize=16)
plt.ylabel(r"$\theta$ [rad]", fontsize=16)
plt.suptitle(r"Sequential Training Predicted Trajectory", fontsize=20)
plt.grid(True)

plt.tight_layout()
plt.subplots_adjust(hspace=0.54,top=0.9)  # Increased spacing between subplots

# Plot State error Vs time for each state
plt.figure(figsize=(10, 8))
plt.subplot(3, 1, 1)
plt.plot(t[num_train:], X[num_train:, 0] - mean_trajectory_random[:, 0], 'b-', label='Random x error')
plt.plot(t[num_train:], X[num_train:, 0] - mean_trajectory_sequential[:, 0], 'r--', label='Sequential x error')
plt.legend(fontsize=14)
plt.title(r"State Error $x$", fontsize=18)
plt.xlabel("Time (s)", fontsize=16)
plt.ylabel("Error", fontsize=16)
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t[num_train:], X[num_train:, 1] - mean_trajectory_random[:, 1], 'b-', label='Random y error')
plt.plot(t[num_train:], X[num_train:, 1] - mean_trajectory_sequential[:, 1], 'r--', label='Sequential y error')
plt.legend(fontsize=14)
plt.title(r"State Error $y$", fontsize=18)
plt.xlabel("Time (s)", fontsize=16)
plt.ylabel("Error", fontsize=16)
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t[num_train:], X[num_train:, 2] - mean_trajectory_random[:, 2], 'b-', label='Random theta error')
plt.plot(t[num_train:], X[num_train:, 2] - mean_trajectory_sequential[:, 2], 'r--', label='Sequential theta error')
plt.legend(fontsize=14)
plt.title(r"State Error $\theta$", fontsize=18)
plt.xlabel("Time (s)", fontsize=16)
plt.ylabel("Error", fontsize=16)
plt.grid(True)

plt.tight_layout()
plt.subplots_adjust(hspace=0.54,top=0.9)  # Increased spacing between subplots
plt.show()