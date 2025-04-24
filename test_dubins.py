import numpy as np
#import matplotlib.pyplot as plt
from vehicle import DubinsCar
from environment import UnboundedPlane

if __name__ == "__main__":
    # Simulation parameters
    dt = 0.1
    steps = 500
    
    # Define theta dictionary for the Dubins car
    theta = {
        "Gamma": np.zeros((3, 3)),  # Process noise (optional)
        "Sigma": np.array([[0.01, 0, 0],
                          [0, 0.01, 0],
                          [0, 0, 0.001]]),  # Measurement noise
        "mu0": np.array([0, 0, 0]),  # Start at origin facing right
        "V0": np.array([[0.1, 0, 0],
                        [0, 0.1, 0],
                        [0, 0, 0.1]])  # Initial state uncertainty
    }

    # Create vehicle
    vehicle = DubinsCar(theta)

    # Define a simple circular motion policy
    class CircularPolicy:
        def __init__(self, v=0.5, omega=0.2):
            self.v = v  # Constant forward velocity
            self.omega = omega  # Constant angular velocity
            
        def get_action(self, s, t):
            return np.array([self.v, self.omega])
    
    policy = CircularPolicy()

    # Create environment
    environment = UnboundedPlane(steps, dt, vehicle, policy)

    # Run simulation
    X, t, U, _ = environment.epoch(animate=True, plot_states=True)

    # Print information about the trajectories
    vehicle.print_history_info()

    plt.show() 