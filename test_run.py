import numpy as np
import matplotlib.pyplot as plt
from vehicle import MassSpringDamper
from policies import NoControl
from environment import UnboundedPlane

if __name__ == "__main__":
    # Load the policy
    dt = 0.2
    steps = 1000
    data_size = 100
    vel_bound = 0.3

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
    # s0 = np.array([1, 2, np.random.uniform(-vel_bound, vel_bound), np.random.uniform(-vel_bound, vel_bound), 0, 0])
    s0 = np.array([1, 2, vel_bound, vel_bound, 0, 0])
    vehicle = MassSpringDamper(s0, p)
    policy = NoControl()

    # Load the environment
    environment = UnboundedPlane(steps, dt, vehicle, policy)
    X, t, U, collision_flag = environment.epoch(animate=True)