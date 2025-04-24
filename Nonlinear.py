import numpy as np
import matplotlib.pyplot as plt
from GP_Train import defineKernel, trainGP, predictGP, split_data
# Load In Dubins Car Data
data = np.load('data/dubins_circular_motion.npz')
X = data['X_set'][:,:,0]  # States
U = data['U_set'][:,:,0]  # Controls
t = data['t_set'][:,0]    # Time vector

T = U.shape[0]
Nx = X.shape[1] 
Nu = U.shape[1] 
sigma_n = 1e-6

# Split data at index 200
(X_train, U_train), (X_test, U_test) = split_data(data, traj_idx=0, split_index=200)

# Extract time for training and test data
# Note: X_train has one more element than U_train (initial state + states after each control)
t_train = t[:X_train.shape[0]]  # Use X_train.shape[0] to get all states including initial
t_test = t[X_train.shape[0]-1:]  # Start from the last training state

model = trainGP(X, U)

x = X_test[0]  # Initial test state
estimates = [x.copy()]

for i in range(U_test.shape[0]):
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
plt.plot(t_test, estimates[0:, 1], 'r--', label='Predicted y')
plt.legend()
plt.title("True vs. Predicted y Position")
plt.xlabel("Time (s)")
plt.ylabel("y Position")
plt.grid(True)

plt.tight_layout()
plt.show()