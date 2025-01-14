import numpy as np

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

class DirectOptimizer():
  def __init__(self, weights):
    pass

  def update_parameters(self, gradient):
    pass
  
  def update_weights(self, weights, learning_rate):
    pass

  def apply_gradients(self, weights, gradient, learning_rate):
    return weights - learning_rate * gradient

class AdamOptimizer():
  def __init__(self, weights, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
    """
    Handles weight updates using ADAM optimization.
    
    Args:
      weights (array): Weights this optimizer will optimize and learn parameters for.
      beta_1 (float): Momentum decay rate (Defaults to 0.9)
      beta_2 (float): Velocity decay rate (Defaults to 0.999)
      epsilon (float): Small value to prevent division by 0 (Defaults to 1e-8)

    Returns:
      weights (array): Updated weights
    
    https://www.geeksforgeeks.org/adam-optimizer/
    https://arxiv.org/abs/1412.6980
    
    M and V:
    mt = B1 * mt-1 + (1-B1) * gradient
    vt = B2 * vt-1 + (1-B2) * gradient**2
      
    Bias correction:
    mt_hat = mt / (1 - B1^t)
    vt_hat = vt / (1 - B2^t)
    
    Weight update:
    wt = wt-1 - alpha * (mt_hat / (sqrt(vt_hat) + E))
    
    gradient -> Gradient of weights wrt. loss
    epsilon  -> Small value to avoid division by 0 (1e-8)
    B1 & B2  -> Decay rates of m, v respectively (B1 = 0.9, B2 = 0.999)
    alpha    -> Learning rate
    """
    self.t = 0
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon

    # Initialize momentum and velocity m, v based on input
    self.m = np.zeros(shape=weights.shape)
    self.v = np.zeros(shape=weights.shape)

  def update_parameters(self, gradient):
    """Update m, v based on mt-1, vt-1 and gradients"""
    # Increment time step
    self.t += 1

    # Update m, v
    self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradient
    self.v = self.beta_2 * self.v + (1 - self.beta_2) * (gradient ** 2)

    # Correct initialization bias of m, v
    # print(f"DEBUG: {self.beta_2**self.t}")
    self.m = self.m / (1 - self.beta_1**self.t + self.epsilon)
    self.v = self.v / (1 - self.beta_2**self.t + self.epsilon)
  
  def update_weights(self, weights, learning_rate):
    """Update w based on m, v and learning rate"""
    weights -= learning_rate * (self.m / (np.sqrt(self.v) + self.epsilon))
    return weights
  
  def apply_gradients(self, weights, gradient, learning_rate):
    """Update parameters and weights, returning updated weights"""
    self.update_parameters(gradient) # update m, v from gradient
    return self.update_weights(weights, learning_rate)