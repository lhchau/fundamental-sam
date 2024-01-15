import math

class Diminish2:
    def __init__(self, optimizer, learning_rate: float):
        self.optimizer = optimizer
        self.base = learning_rate
        self.epoch = 0

    def step(self):
        epoch = self.epoch
        lr = self.base * 1 / ((epoch+1) ** 0.5001)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        
        self.epoch += 1

    def get_last_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]