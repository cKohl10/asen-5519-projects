# This script will generate a dataset of states over time for a given policy

import numpy as np
from environment import UnboundedPlane
from vehicle import MassSpringDamper
from policies import StepPolicy

if __name__ == "__main__":
    # Load the policy
    s0 = np.array([0, 0, 0, 0])
    dt = 0.1
    steps = 1000
    p = np.array([3, 1, 0.2, 0.5, 1])
    vehicle = MassSpringDamper(s0, p)
    policy = StepPolicy()

    # Load the environment
    environment = UnboundedPlane(steps, dt, vehicle, policy)

    environment.epoch()

