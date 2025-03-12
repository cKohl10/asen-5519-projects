import numpy as np

class Policy:
    def __init__(self):
        pass

    def get_action(self, state):
        pass

class StepPolicy(Policy):
    def __init__(self):
        super().__init__()

    def get_action(self, s):
        u = np.zeros(2)
        u[1] = 1
        return u

