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
                length_scale=10.0,
                sigma_f=0.1,
                sigma_n=0.01)

# # Parameter grid
# length_scales = [2.0, 5.0,10.0]
# sigma_fs = [0.1, 0.5, 1.0, 2.0, 5.0]
# sigma_ns = [0.01, 0.1, 0.5, 1.0]

# best_rmse = float('inf')
# best_params = None
# best_model = None

# print("Optimizing parameters...")
# for length_scale in tqdm(length_scales, desc="Length scales"):
#     for sigma_f in sigma_fs:
#         for sigma_n in sigma_ns:
#             rmse, model = evaluate_model(X, U, length_scale, sigma_f, sigma_n)
#             total_rmse = np.mean(rmse)
#             print(f"Testing - Length scale: {length_scale}, Sigma f: {sigma_f}, Sigma n: {sigma_n}, RMSE: {total_rmse}")
#             if total_rmse < best_rmse:
#                 best_rmse = total_rmse
#                 best_params = (length_scale, sigma_f, sigma_n)
#                 best_model = model
#                 print(f"\nNew best parameters found:")
#                 print(f"Length scale: {length_scale}")
#                 print(f"Sigma f: {sigma_f}")
#                 print(f"Sigma n: {sigma_n}")
#                 print(f"RMSE: {total_rmse}")

# print("\nBest parameters found:")
# print(f"Length scale: {best_params[0]}")
# print(f"Sigma f: {best_params[1]}")
# print(f"Sigma n: {best_params[2]}")
# print(f"Best RMSE: {best_rmse}")

# Plot results with best model
plt.figure(figsize=(12, 8))
# Plot x position
plt.subplot(3, 1, 1)
plt.plot(t, X[:, 0], 'b-', label='True x')
plt.plot(t[200:], model['sampled_trajectory'][:, 0], 'r--', label='Predicted x')
# Plot 2-sigma bounds as filled region
# plt.fill_between(t, 
#                 best_model['sampled_trajectory'][1:, 0] - 2*np.sqrt(best_model['sampled_variances'][:, 0]),
#                 best_model['sampled_trajectory'][1:, 0] + 2*np.sqrt(best_model['sampled_variances'][:, 0]),
#                 color='g', alpha=0.2, label=r'2$\sigma$ bounds')
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
# plt.fill_between(t,
#                 best_model['sampled_trajectory'][1:, 1] - 2*np.sqrt(best_model['sampled_variances'][:, 1]),
#                 best_model['sampled_trajectory'][1:, 1] + 2*np.sqrt(best_model['sampled_variances'][:, 1]),
#                 color='g', alpha=0.2, label=r'2$\sigma$ bounds')
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
# plt.fill_between(t,
#                 best_model['sampled_trajectory'][1:, 2] - 2*np.sqrt(best_model['sampled_variances'][:, 2]),
#                 best_model['sampled_trajectory'][1:, 2] + 2*np.sqrt(best_model['sampled_variances'][:, 2]),
#                 color='g', alpha=0.2, label=r'2$\sigma$ bounds')
plt.legend()
plt.title(r"True vs. Predicted $\theta$")
plt.xlabel("Time (s)")
plt.ylabel(r"$\theta$ [rad]")
plt.grid(True)

plt.tight_layout()
plt.show()
