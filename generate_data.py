# This script will generate a dataset of states over time for a given policy

import numpy as np
from environment import UnboundedPlane
from vehicle import MassSpringDamper
from policies import StepPolicy, NoControl, SinePolicy

if __name__ == "__main__":
    # Load the policy
    s0 = np.array([1, 2, 0, 0, 0, 0])
    dt = 0.2
    steps = 1000

    # Parameters for the system
    # m1, m2, k1, k2, b, r, cor
    m1 = 10 #Kg
    m2 = 20 #Kg
    k1 = 5 #N/m
    k2 = 5 #N/m
    b = 5 #N*s/m
    r = 0.3 #m
    cor = 0.2 #[-]
    p = np.array([m1, m2, k1, k2, b, r, cor])
    vehicle = MassSpringDamper(s0, p)
    policy = SinePolicy(w=0.5, A=2)

    # Load the environment
    environment = UnboundedPlane(steps, dt, vehicle, policy)

    environment.epoch()

