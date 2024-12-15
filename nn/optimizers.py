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
        
class LinearCycleSchedule(LearningRateSchedule):
    def __init__(self, min_lr, max_lr, cycle_rate):
        # Set initial lr to minimum
        self.lr = min_lr
        
        # Set bounds
        self.min = min_lr
        self.max = max_lr
        
        # Set linear increase/decrease rate based on phase
        self.rate = cycle_rate
        self.phase = 1
        
    def __call__(self):
        self.increment()
        
        # If learning rate hits max, inverse phase and return max
        if self.lr >= self.max:
            self.phase = -1
            return self.max
        
        # If learning rate hits min, inverse phase and return min
        if self.lr <= self.min:
            self.phase = 1
            return self.min
        
        return self.lr
    
    def increment(self):
        # Update lr positively or negatively by rate based on phase
        self.lr *= 1 + (self.phase*self.rate)