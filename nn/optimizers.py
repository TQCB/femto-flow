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
  def __init__(self, weights, beta_1=0.9, beta_2=0.999, epsilon=10e-8):
    """
    Handles weight updates using ADAM optimization.
    
    Args:
      weights (array): Weights this optimizer will optimize and learn parameters for.
      beta_1 (float): Momentum decay rate (Defaults to 0.9)
      beta_2 (float): Velocity decay rate (Defaults to 0.999)
      epsilon (float): Small value to prevent division by 0 (Defaults to 1e-8)
    
    https://www.geeksforgeeks.org/adam-optimizer/
    https://arxiv.org/abs/1412.6980
    
    M and V:
    mt = B1 * mt-1 + (1-B1) * gradient
    vt = B2 * vt-1 + (1-B2) * gradient**2
      
    Bias correction:
    mt_hat = mt / (1 - B1^t)
    vt_hat = vt / (1 - B2^t)
    
    Weight update:
    wt = wt-1 - mt_hat * (alpha / (sqrt(vt_hat) + E))
    
    gradient -> Gradient of weights wrt. loss
    epsilon  -> Small value to avoid division by 0 (1e-8)
    B1 & B2  -> Decay rates of m, v respectively (B1 = 0.9, B2 = 0.999)
    alpha    -> Learning rate
    """
    raise NotImplementedError

  def update(self, ):
    """Update m, v based on mt-1, vt-1 and gradients"""
    raise NotImplementedError
  
  def apply_gradients(self):
    """Update w based on m, v and learning rate"""
    raise NotImplementedError