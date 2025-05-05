import numpy as np
from tqdm import tqdm
from environment import UnboundedPlane
from vehicle import DubinsCar
from policies import NoControl, SinePolicy

def generate_data(policy, vehicle, sim_params, save_path=None, save_name=None):
    # 3 states (x, y, θ)
    steps = sim_params["steps"]
    dt = sim_params["dt"]
    data_size = sim_params["data_size"]
    Nu = vehicle.Nu
    Nx = vehicle.Nx
    X_set = np.zeros((steps+1, Nx, data_size))  # Noisy data
    t_set = np.zeros((steps+1, data_size))
    U_set = np.zeros((steps, Nu, data_size))  # One less control than steps

    i = 0
    pbar = tqdm(total=data_size, desc="Generating Dubins car data")
    while i < data_size:
        vehicle.reset()
        environment = UnboundedPlane(steps, dt, vehicle, policy)

        # Animate only first trajectory
        if i == 0:
            X, t, U, _ = environment.epoch(animate=True)
        else:
            X, t, U, _ = environment.epoch()

        # Unwrap list of states into a single array
        X = np.array(X)
        X_set[:, :, i] = X
        t_set[:, i] = t
        U_set[:, :, i] = U

        pbar.update(1)
        i += 1

    pbar.close()

    if save_path is not None:
        np.savez(save_path + save_name, X_set=X_set, t_set=t_set, U_set=U_set)
    return X_set, t_set, U_set

if __name__ == "__main__":
    # Simulation parameters
    dt = 0.1
    steps = 500
    data_size = 1
    
    # Initial state uncertainty
    pos_bound = 1.0  # Random initial position within ±1m
    theta_bound = np.pi/6  # Random initial heading within ±30°

    # Noise parameters
    noise_level = 0.09  # Measurement noise

    # Define theta dictionary for the Dubins car
    theta = {
        "A": np.zeros((3, 3)),  # Not used for nonlinear dynamics
        "Gamma": np.zeros((3, 3)),  # No process noise
        "C": np.eye(3),  # Observe all states
        "Sigma": np.array([[noise_level/10, 0, 0],
                          [0, noise_level/10, 0],
                          [0, 0, noise_level/100]]), 
        "mu0": np.array([0, 0, 0]),  # Start at origin facing right
        "V0": np.array([[pos_bound, 0, 0],
                        [0, pos_bound, 0],
                        [0, 0, theta_bound]]),  # Initial state uncertainty
        "B": np.zeros((3, 2))  # Not used for nonlinear dynamics
    }

    # Create vehicle and policy
    vehicle = DubinsCar(theta)
    
    # Example policies:
    # 1. No control (vehicle stays still)
    policy_no_control = NoControl()
    
    # 2. Circular motion
    v = 0.5  # Constant velocity
    omega = 0.2  # Constant angular velocity
    class CircularPolicy:
        def get_action(self, s, t):
            return np.array([v, omega])
    
    policy = CircularPolicy()
    
    # 3. Random motion
    class RandomPolicy:
        def __init__(self, v_min=-1.0, v_max=1.0, omega_min=-np.pi, omega_max=np.pi):
            self.v_min = v_min
            self.v_max = v_max
            self.omega_min = omega_min
            self.omega_max = omega_max
            
        def get_action(self, s, t):
            # Generate random control inputs
            v = np.random.uniform(self.v_min, self.v_max)
            omega = np.random.uniform(self.omega_min, self.omega_max)
            return np.array([v, omega])
    
    # Uncomment to use random policy instead of circular
    #policy = RandomPolicy()

    sim_params = {
        "steps": steps,
        "dt": dt,
        "data_size": data_size,
        "allow_collision": True  # No collision checking for Dubins car
    }

    # Generate and save data
    X_set, t_set, U_set = generate_data(
        policy, 
        vehicle, 
        sim_params, 
        save_path="data/", 
        save_name="dubins_circular_motion.npz"
    )
