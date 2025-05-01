import numpy as np
import matplotlib.pyplot as plt
from GP_Train import defineKernel, trainGP_V2
from tqdm import tqdm

print("Loading training data...")
circ_data = np.load('data/dubins_circular_motion.npz')

X = circ_data['X_set'][:,:,0]
U = circ_data['U_set'][:,:,0]
t = circ_data['t_set'][:,0]

# Train GP model
model = trainGP_V2(X, U, 
                training_points=300, 
                sampling_steps=200, 
                length_scale = 10, 
                sigma_f=0.127, 
                sigma_n=7.84*10**-5)

# Extract sampled trajectory and append to training data
X_sampled = model['sampled_trajectory']
X_sampled = np.concatenate((X[300:], X_sampled), axis=0)

# Extract sampled controls and append to training data
U_sampled = model['sampled_controls']
U_sampled = np.concatenate((U[300:], U_sampled), axis=0)

# Extract variances 
variances = model['sampled_variances']

# Display the sampled trajectory matrix
# print(X_sampled)
# print(U_sampled)
print(variances)

# Plot True Trajectory and GP Prediction in subplots for each state variable
plt.figure(figsize=(12, 8))
# Plot x position
plt.subplot(3, 1, 1)
#plt.plot(t, X[:, 0], 'b-', label='True x')
plt.plot(t[300:], model['sampled_trajectory'][:, 0], 'r--', label='Predicted x')
plt.plot(t[301:], model['sampled_trajectory'][1:, 0] - 10*np.sqrt(variances[:, 0]), 'g--', label=r'2$\sigma$ bounds')
plt.plot(t[301:], model['sampled_trajectory'][1:, 0] + 10*np.sqrt(variances[:, 0]), 'g--')
plt.legend()
plt.title(r"True vs. Predicted $x$ Position")
plt.xlabel("Time (s)")
plt.ylabel("$x$ Position")
plt.grid(True)

# Plot y position
plt.subplot(3, 1, 2)
#plt.plot(t, X[:, 1], 'b-', label='True y')
# Start plotting estimates at the end of training data
plt.plot(t[300:], model['sampled_trajectory'][:, 1], 'r--', label='Predicted y')
plt.plot(t[301:], model['sampled_trajectory'][1:, 1] - 10*np.sqrt(variances[:, 1]), 'g--', label=r'2$\sigma$ bounds')
plt.plot(t[301:], model['sampled_trajectory'][1:, 1] + 10*np.sqrt(variances[:, 1]), 'g--')
plt.legend()
plt.title(r"True vs. Predicted $y$ Position")
plt.xlabel("Time (s)")
plt.ylabel("$y$ Position")
plt.grid(True)

# Plot theta
plt.subplot(3, 1, 3)
#plt.plot(t, np.mod(X[:, 2], 2*np.pi), 'b-', label=r'True $\theta$')
# Start plotting estimates at the end of training data
plt.plot(t[300:], np.mod(model['sampled_trajectory'][:, 2], 2*np.pi), 'r--', label=r'Predicted $\theta$')
plt.plot(t[301:], np.mod(model['sampled_trajectory'][1:, 2] - 10*np.sqrt(variances[:, 2]), 2*np.pi), 'g--', label=r'2$\sigma$ bounds')
plt.plot(t[301:], np.mod(model['sampled_trajectory'][1:, 2] + 10*np.sqrt(variances[:, 2]), 2*np.pi), 'g--')
plt.legend()
plt.title(r"True vs. Predicted $\theta$")
plt.xlabel("Time (s)")
plt.ylabel(r"$\theta$ [rad]")
plt.grid(True)

plt.tight_layout()
plt.show()


# plt.figure()
# plt.plot(X_sampled[:, 0], X_sampled[:, 1])  # x vs y
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Sampled Trajectory')
# plt.show()
