from vehicle import MassSpringDamper, DubinsCar
from policies import NoControl
from environment import UnboundedPlane
from train_EM import plot_data
from scipy.linalg import expm
import numpy as np
import matplotlib.pyplot as plt

def convert_lin_to_pred_dyn(theta, dt):
    # Convert continuous linear dynamics to discrete dynamics
    A = expm(theta["A"]*dt)

    theta["A"] = A

    return theta

if __name__ == "__main__":
    # Load the data
    try:
        data = np.load("data/noisy_data.npz", allow_pickle=True)
    except:
        data = np.load("asen-5519-projects/data/noisy_data.npz", allow_pickle=True)

    t_set = data["t_set"]
    dt = t_set[2,0] - t_set[1,0]
    steps = len(t_set)  

    theta = data["theta"]
    theta = theta.item()

    theta = convert_lin_to_pred_dyn(theta, dt)

    vehicle = MassSpringDamper(theta)
    policy = NoControl()
    environment = UnboundedPlane(steps, dt, vehicle, policy, bounds=np.array([100, 100]))
    X, t, U, collision_flag = environment.epoch(animate=False, use_dynamics=False)
    pred_data = {"X_set": np.array(X), "t_set": np.array(t), "U_set": np.array(U)}

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig = plot_data(axes, data, pred_data)
    plt.show()