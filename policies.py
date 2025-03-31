import numpy as np

class Policy:
    def __init__(self):
        pass

    def get_action(self, state):
        pass

class NoControl(Policy):
    def __init__(self):
        super().__init__()

    def get_action(self, s, t):
        return np.zeros(2)

class StepPolicy(Policy):
    def __init__(self):
        super().__init__()

    def get_action(self, s, t):
        u = np.zeros(2)
        u[0] = 1
        return u
    
class SinePolicy(Policy):
    def __init__(self, w, A):
        super().__init__()
        self.w = w
        self.A = A

    def get_action(self, s, t):
        u = np.zeros(2)
        u[0] = self.A * np.sin(self.w * t)
        return u

class StateReferenceFeedbackPolicy(Policy):
    def __init__(self, Kp, F, s_ref, tracked_indices=[0, 1]):
        super().__init__()
        self.Kp = Kp
        self.F = F
        self.s_ref = s_ref
        self.tracked_indices = tracked_indices

    def get_action(self, s, t):
        u = self.F @ self.s_ref - self.Kp @ s
        return u

