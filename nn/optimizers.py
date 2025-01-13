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

class Adam:
  def __init__(self):
    """
    https://www.geeksforgeeks.org/adam-optimizer/
    https://arxiv.org/abs/1412.6980
    
    - Needs its own weights (m & v)
    - Needs to update weights at each iteration
    - Needs to apply weights at each iteration
    - Needs to take an LR object
    
    M and V:
    mt = B1 * mt-1 + (1-B1) * gradient
    vt = B2 * vt-1 + (1-B1) * gradient**2\
      
    Bias correction
    mt_hat = mt / (1 - B1)
    vt_hat = vt / (1 - B2)
    
    Weight update:
    wt = wt-1 - mt * (alpha / (sqrt(vt) + E))
    
    gradient-> Gradient of weights wrt. loss
    epsilon -> Small value to avoid division by 0 (1e-8)
    B1 & B2 -> Decay rates of m, v respectively (B1 = 0.9, B2 = 0.999)
    alpha   -> Learning rate
    
    Example:
    $ optimiser.apply_gradients(weights, gradients, lr)
    
    > apply_gradients(...):
    >   update(gradients) # use gradients to update m, v
    >   return 
    
    """
    raise NotImplementedError

  def update(self, ):
    """Update m, v based on mt-1, vt-1 and gradients"""
    raise NotImplementedError
  
  def apply_gradients(self):
    """Update w based on m, v and learning rate"""
    raise NotImplementedError