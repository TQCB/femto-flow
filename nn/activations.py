import numpy as np

class TanH:
  def forward(self, x):
    return np.tanh(x)
    
  def backward(self, x):
    return 1-np.tanh(x)**2

class Sigmoid:
  def forward(self, x):
    return 1 / (1 + np.exp(-x))
  
  def backward(self, x):
      return 1 / (1 + np.exp(-x)) * (1 - (1 / (1 + np.exp(-x))))

class Softmax:
  def forward(self, x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # numerically stable
    return e_x / np.sum(e_x, axis=-1, keepdims=True)
  
  def backward(self, x):
    s = self.forward(x)
    return s * (1 - s)
  
class ReLU:
  def forward(self, x):
    out = np.maximum(0, x)
    return out
  
  def backward(self, x):
    return np.where(x > 0, 1, 0)
  
class Swish:
  def __init__(self, beta=1.0):
    self.beta = beta

  def forward(self, x):
    return x * 1 / (1 + np.exp(- self.beta * x))

  def backward(self, x):
    sigmoid_x = 1 / (1 + np.exp(- self.beta * x))
    return sigmoid_x + x * self.beta * sigmoid_x * (1 - sigmoid_x)