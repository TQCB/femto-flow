class LearningRateSchedule:
    def __init__(self, initial_lr):
        self.lr = initial_lr
    
    def __call__(self):
        self.increment()
        return self.lr
    
    def increment(self):
        pass

class ExponentialDecaySchedule(LearningRateSchedule):
    def __init__(self, initial_lr, decay_steps, decay_rate, min=0):
        self.n_step = 0
        self.lr = initial_lr
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.min = min
    
    def __call__(self):
        self.n_step += 1
        
        if (self.n_step >= self.decay_steps) &\
            (self.lr > self.min):
            self.increment()
            self.n_step = 0
            
        return self.lr
    
    def increment(self):
        self.lr *= self.decay_rate