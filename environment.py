# Environment Definitions
import matplotlib.pyplot as plt

class Environment:
    def __init__(self):
        pass

    def reset(self):
        pass

    def step(self, policy):
        pass

    def epoch(self, policy):
        pass

class UnboundedPlane(Environment):
    def __init__(self, steps, dt, vehicle, policy):
        super().__init__()
        self.steps = steps
        self.dt = dt
        self.t = 0
        self.vehicle = vehicle
        self.policy = policy

    def reset(self):
        self.vehicle.reset()

    def step(self, policy):
        self.vehicle.step(policy, self.dt, self.t)
        self.t += self.dt

    def epoch(self, animate=False, plot_states=False):
        for _ in range(self.steps):
            self.step(self.policy)

        if animate:
            self.vehicle.plot_states()
            ani = self.vehicle.animate()
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
    


