# Vehicle Definitions
import numpy as np

class Vehicle:
    def __init__(self):
        pass

    def reset(self):
        pass

    def step(self, policy):
        pass

class DubinsUnicycle(Vehicle):
    def __init__(self):
        super().__init__()
        self.state = np.array([0, 0, 0])
        self.dt = 0.1

        # Underlying Dynamics
        self.max_speed = 1.0

    def reset(self):
        self.state = np.array([0, 0, 0])

    def step(self, policy):
        pass
