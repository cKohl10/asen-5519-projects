# Environment Definitions
import matplotlib.pyplot as plt
import numpy as np
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
    


