# Trainers for learning the vehicle dynamics

class Trainer:
    def __init__(self, vehicle, policy, data_loader):
        self.vehicle = vehicle
        self.policy = policy
        self.data_loader = data_loader

    def train(self):
        pass

class MonteCarloTrainer(Trainer):
    def __init__(self, vehicle, policy, data_loader):
        super().__init__(vehicle, policy, data_loader)

    def train(self):
        pass




