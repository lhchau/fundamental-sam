import math

class Diminish3:
    def __init__(self, optimizer, learning_rate: float):
        self.optimizer = optimizer
        self.base = learning_rate
        self.epoch = 0
        self.base_step = 2
        self.prev_lr = self.base * 1 / (self.base_step * math.log(self.base_step))

    def step(self):
        epoch = self.epoch
        if epoch % 5 == 0:
            self.base_step += 1
            lr = self.base * 1 / (self.base_step * math.log(self.base_step))
            self.prev_lr = lr
        else:
            lr = self.prev_lr

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        
        self.epoch += 1

    def get_last_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]