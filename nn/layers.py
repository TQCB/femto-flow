import numpy as np

################################################################################

# Module containing definitions of different layers for our neural networks.
# Layers at a minimum define forward and backward pass methods. These 
# respectively apply the layer's calculations to an input, passing the result to
# the next layer and propagating the error to the previous layer.

# TODO
#  |- HIGH
#      |- Embedding
#      |- Attention
#  |- NORMAL
#      |- Recurrent
#      |- Convolutional 

################################################################################

class Layer:
  def __init__(self, input_dim, output_dim):
    self.weights = np.random.uniform(-1, 1, size=(input_dim, output_dim))
    self.bias = np.random.uniform(-1, 1,size=output_dim) - 0.5
    
  # compute h(X) = Y
  def forward_propagation(self, input):
    self.input = input
    self.output = np.dot(self.input, self.weights) + self.bias
    return self.output

  # compute dE/dX for dE/dY and update dX
  def backward_propagation(self, output_error, learning_rate):
    input_error = np.dot(output_error, self.weights.T)
    weights_error = np.dot(self.input.T, output_error)
        
    self.weights -= learning_rate * weights_error
    self.bias -= learning_rate * output_error
        
    return input_error
    
class ActivationLayer(Layer):
  def __init__(self, activation):
    self.activation = activation()

  # compute h(X) = Y
  def forward_propagation(self, input):
    self.input = input
    self.output = self.activation.forward(self.input)
    return self.output
    
    # compute dE/dX for dE/dY 
  def backward_propagation(self, output_error, learning_rate):
    return self.activation.backward(self.input) * output_error
    
class FCLayer(Layer):
  def __init__(self, input_dim, output_dim):
    self.weights = np.random.rand(input_dim, output_dim) - 0.5
    self.bias = np.random.rand(1, output_dim) - 0.5
        
  def forward_propagation(self, input):
    self.input = input
    self.output = np.dot(self.input, self.weights) + self.bias
    return self.output
    
  def backward_propagation(self, output_error, learning_rate):
    input_error = np.dot(output_error, self.weights.T)
    weights_error = np.dot(self.input.T, output_error)
        
    self.weights -= learning_rate * weights_error
    self.bias -= learning_rate * output_error
    return input_error
    
class MultiHeadAttention(Layer):
  """
  Multi Head Self Attention for Dense networks.
    
  Args:
        
  """
    
  def __init__(self, input_dim, n_dim, n_heads):
        
    self.n_heads = n_heads
    self.n_dim = n_dim
    self.head_dim = n_dim // n_heads