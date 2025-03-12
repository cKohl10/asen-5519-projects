# Environment Definitions

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
        self.vehicle = vehicle
        self.policy = policy

    def reset(self):
        self.vehicle.reset()

    def step(self, policy):
        self.vehicle.step(policy, self.dt)

    def epoch(self):
        for _ in range(self.steps):
            self.step(self.policy)

        self.vehicle.plot_states()
        self.vehicle.animate()
    


