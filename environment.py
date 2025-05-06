# Environment Definitions
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from utils.common import cont2disc_AQ

class Environment:
    def __init__(self):
        pass

    def reset(self):
        pass

    def step(self, policy):
        pass

    def epoch(self, policy):
        pass

    def train(self):
        pass

class SimpleEnv(Environment):
    def __init__(self, steps, dt, vehicle, policy, bounds=None):
        super().__init__()
        self.steps = steps
        self.dt = dt
        self.t = 0
        self.vehicle = vehicle
        self.policy = policy

    def reset(self):
        self.vehicle.reset()
        self.t = 0

    def step(self, policy, discrete=True):
        if discrete:
            self.vehicle.discrete_step(policy, self.dt, self.t)
        else:
            self.vehicle.continuous_step(policy, self.dt, self.t)
        self.t = self.vehicle.current_time

    def epoch(self, animate=False, plot_states=False, discrete=True, save_path=None):
        for _ in range(self.steps-1):
            self.step(self.policy, discrete)

        if animate:
            self.vehicle.plot_states()
            ani = self.vehicle.animate(save_path=save_path)
            plt.show(block=False)

        if plot_states:
            self.vehicle.plot_states()
            plt.show(block=False)
            
        Z = self.vehicle.state_history
        X = self.vehicle.obs_history
        t = self.vehicle.time_history
        U = self.vehicle.control_history

        return Z, X, t, U
    
    def generate_data(self, iterations, save_path=None, save_name=None, animate=True, discrete=True, plot_states=True, nyquist_freq=None):
        # Case where the system is localized by a "CV" algorithm with noise and only the position is observed
        # Data will be in the size of (Nx, steps, data_size)
        data_size = iterations
        X_set = np.zeros((self.steps, self.vehicle.theta.Nx, data_size)) # Noisy data
        Z_set = np.zeros((self.steps, self.vehicle.theta.Ns, data_size)) # Noisy data
        t_set = np.zeros((self.steps, data_size))
        U_set = np.zeros((self.steps-1, self.vehicle.theta.Nu, data_size)) # One less control than steps to account for initial state

        if nyquist_freq is not None:
            self.dt = self.dt / nyquist_freq
            self.steps = self.steps * nyquist_freq

        i = 0
        pbar = tqdm(total=data_size, desc="Generating data")
        while i < data_size:
            self.reset()

            # Load the environment
            if i == 0:
                # X, t, U, collision_flag = environment.epoch(animate=True, save_path=f"animations/mass_spring_damper_{save_name}.gif")
            
                Z, X, t, U = self.epoch(animate=animate, plot_states=plot_states, discrete=discrete)
            else:
                Z, X, t, U = self.epoch(discrete=discrete)

            # Unwrap list of states into a single array
            Z = np.array(Z)
            X = np.array(X)
            U = np.array(U)
            
            # Get every nyquist_freq frame if specified
            if nyquist_freq is not None:
                Z = Z[::nyquist_freq]
                X = X[::nyquist_freq]
                t = t[::nyquist_freq]
                U = U[::nyquist_freq]
                if len(U) > np.shape(U_set)[0]:
                    U = U[:np.shape(U_set)[0]]
            
            Z_set[:, :, i] = Z
            X_set[:, :, i] = X
            t_set[:, i] = t
            U_set[:, :, i] = U

            pbar.update(1)
            i += 1

        pbar.close()

        theta = self.vehicle.get_theta()
        if not discrete:
            theta.A, theta.Gamma = cont2disc_AQ(theta.A, theta.Gamma, self.dt)

        if save_path is not None:
            np.savez(save_path + save_name + ".npz", Z_set=Z_set, X_set=X_set, t_set=t_set, U_set=U_set, theta=theta)

        return Z_set, X_set, t_set, U_set

class UnboundedPlane(Environment):
    def __init__(self, steps, dt, vehicle, policy, bounds=None):
        super().__init__()
        self.steps = steps
        self.dt = dt
        self.t = 0
        self.vehicle = vehicle
        self.policy = policy
        if bounds is None:
            self.bounds = np.array([100, 100])
        else:
            self.bounds = bounds

    def reset(self):
        self.vehicle.reset()

    # def step(self, policy):
    #     self.vehicle.step(policy, self.dt, self.t)
    #     self.t += self.dt

    def epoch(self, animate=False, plot_states=False, use_dynamics=True, save_path=None):
        for _ in range(self.steps):
            if use_dynamics:
                self.vehicle.step(self.policy, self.dt, self.t)
            else:
                self.vehicle.prediction_step(self.policy, self.dt, self.t)
            self.t += self.dt

            if self.vehicle.s[0] > self.bounds[0] or self.vehicle.s[0] < -self.bounds[0] or self.vehicle.s[1] > self.bounds[1] or self.vehicle.s[1] < -self.bounds[1]:
                print("Out of bounds. Unstable?")
                break

        if animate:
            self.vehicle.plot_states()
            ani = self.vehicle.animate(save_path=save_path)
            plt.show()

        if plot_states:
            self.vehicle.plot_states()
            plt.show()

        X = self.vehicle.state_history
        t = self.vehicle.time_history
        U = self.vehicle.control_history
        collision_flag = self.vehicle.collision_flag

        # if self.vehicle.collision_flag:
        #     print("Collision detected")
            

        return X, t, U, collision_flag
    


