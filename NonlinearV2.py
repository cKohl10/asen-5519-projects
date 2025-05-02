import numpy as np
import matplotlib.pyplot as plt
from GP_TrainV2 import defineKernel, trainGP_V2
from tqdm import tqdm

print("Loading training data...")
circ_data = np.load('data/dubins_circular_motion.npz')

X = circ_data['X_set'][:,:,0]
U = circ_data['U_set'][:,:,0]
t = circ_data['t_set'][:,0]

# Train GP model with adjusted parameters
model = trainGP_V2(X, U, 
                training_points=200, 
                sampling_steps=300, 
                length_scale = 10.0,
                sigma_f=1.0,
                sigma_n=1e-6)


# Plot True Trajectory and GP Prediction in subplots for each state variable
plt.figure(figsize=(12, 8))
# Plot x position
plt.subplot(3, 1, 1)
plt.plot(t, X[:, 0], 'b-', label='True x')
plt.plot(t[200:], model['sampled_trajectory'][:, 0], 'r--', label='Predicted x')
# Plot 2-sigma bounds as filled region
plt.fill_between(t[201:], 
                model['sampled_trajectory'][1:, 0] - 2*np.sqrt(model['sampled_variances'][:, 0]),
                model['sampled_trajectory'][1:, 0] + 2*np.sqrt(model['sampled_variances'][:, 0]),
                color='g', alpha=0.2, label=r'2$\sigma$ bounds')
plt.legend()
plt.title(r"True vs. Predicted $x$ Position")
plt.xlabel("Time (s)")
plt.ylabel("$x$ Position")
plt.grid(True)

# Plot y position
plt.subplot(3, 1, 2)
plt.plot(t, X[:, 1], 'b-', label='True y')
plt.plot(t[200:], model['sampled_trajectory'][:, 1], 'r--', label='Predicted y')
# Plot 2-sigma bounds as filled region
plt.fill_between(t[201:],
                model['sampled_trajectory'][1:, 1] - 2*np.sqrt(model['sampled_variances'][:, 1]),
                model['sampled_trajectory'][1:, 1] + 2*np.sqrt(model['sampled_variances'][:, 1]),
                color='g', alpha=0.2, label=r'2$\sigma$ bounds')
plt.legend()
plt.title(r"True vs. Predicted $y$ Position")
plt.xlabel("Time (s)")
plt.ylabel("$y$ Position")
plt.grid(True)

# Plot theta
plt.subplot(3, 1, 3)
plt.plot(t, X[:, 2], 'b-', label=r'True $\theta$')
plt.plot(t[200:], model['sampled_trajectory'][:, 2], 'r--', label=r'Predicted $\theta$')
# Plot 2-sigma bounds as filled region
plt.fill_between(t[201:],
                model['sampled_trajectory'][1:, 2] - 2*np.sqrt(model['sampled_variances'][:, 2]),
                model['sampled_trajectory'][1:, 2] + 2*np.sqrt(model['sampled_variances'][:, 2]),
                color='g', alpha=0.2, label=r'2$\sigma$ bounds')
plt.legend()
plt.title(r"True vs. Predicted $\theta$")
plt.xlabel("Time (s)")
plt.ylabel(r"$\theta$ [rad]")
plt.grid(True)

plt.tight_layout()
plt.show()

