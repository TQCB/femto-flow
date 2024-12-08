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
  def forward(self, input):
    self.input = input
    self.output = np.dot(self.input, self.weights) + self.bias
    return self.output

  # compute dE/dX for dE/dY and update dX
  def backward(self, output_error, learning_rate):
    input_error = np.dot(output_error, self.weights.T)
    weights_error = np.dot(self.input.T, output_error)
        
    self.weights -= learning_rate * weights_error
    self.bias -= learning_rate * output_error
        
    return input_error
    
class ActivationLayer(Layer):
  def __init__(self, activation):
    self.activation = activation()

  # compute h(X) = Y
  def forward(self, input):
    self.input = input
    self.output = self.activation.forward(self.input)
    return self.output
    
    # compute dE/dX for dE/dY 
  def backward(self, output_error, learning_rate):
    return self.activation.backward(self.input) * output_error
    
class FCLayer(Layer):
  def __init__(self, input_dim, output_dim):
    self.weights = np.random.rand(input_dim, output_dim) - 0.5
    self.bias = np.random.rand(1, output_dim) - 0.5
        
  def forward(self, input):
    self.input = input
    self.output = np.dot(self.input, self.weights) + self.bias
    return self.output
    
  def backward(self, output_error, learning_rate):
    input_error = np.dot(output_error, self.weights.T)
    weights_error = np.dot(self.input.T, output_error)
        
    self.weights -= learning_rate * weights_error
    self.bias -= learning_rate * output_error
    return input_error
    
class MultiHeadAttention(Layer):
  """
  Multi Head Self Attention for Dense networks.
    
  Args:
		input dim: Dimension of input (most likely the sequence len)
		n_dim: Total dimensions for Q, K, V matrices
		n_heads: Quantity of attention heads
  """
    
  def __init__(self, input_dim, n_dim, n_heads):
    self.n_heads = n_heads
    self.n_dim = n_dim
    self.head_dim = n_dim // n_heads # dims for qkv per head
    
    # Initialize weights for Q, K, V
    self.wq = np.random.rand(input_dim, self.n_dim) - 0.5
    self.wk = np.random.rand(input_dim, self.n_dim) - 0.5
    self.wv = np.random.rand(input_dim, self.n_dim) - 0.5
    
    # Output projection weights
    self.wo = np.random.rand(self.n_dim, self.n_dim) - 0.5
    
  def split_heads(self, x):
    """Utility function to split global Q, K, V into (n_heads, seq_len, head_dim)"""
    seq_len = x.shape[0]
    return x.reshape(self.n_heads, seq_len, self.head_dim)
    
  def combine_heads(self, x):
    """Utility function to combine head Q, K, V into (seq_len, n_dim)"""
    seq_len = x.shape[1]
    print(f"Combine seqlen: {seq_len}")
    return x.reshape(seq_len, self.n_dim)
  
  def forward(self, input):
    """
    Forward pass through attention layer.
    
    Algo:
			- Initiate QKV and output projection matrices
			- Separate these matrices between the heads
			- Calculate scores: (QK.t)
				- Scale by d**0.5 for gradient stability and normalization WRT d
			- Calculate softmax of scores: exp(scores) / sum(exp(scores))
				- Substract max(scores) from the softmax for numerical stability
			- Calculate attention ouput: softmax(QK.t)V
			- Combine all head outputs
			- Project attention output into original space with output projection matrix
				-softmax(QK.t)VO
    
    """
    self.input = input
    
    # Compute Q, K, V
    # shape: (seq_len, output_dim)
    Q = np.dot(input, self.wq)
    K = np.dot(input, self.wk)
    V = np.dot(input, self.wv)
    
    # Split Q, K, V into heads
    # shape: (n_heads, seq_len, head_dim)
    Q = self.split_heads(Q)
    K = self.split_heads(K)
    V = self.split_heads(V)
    
    # Compute Luong attention (scaled dot product)
    # Divide by sqrt(d) for more stable gradient flow
    # Normalizes the magnitude of score with respect to d
    # shape: (n_heads, seq_len, seq_len)
    # score of each Q to each K, each element of sequence to every other
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(self.head_dim)
    
    # Softmax scores
    # shape: (n_heads, seq_len, seq_len)
    attention_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights /= np.sum(attention_weights, axis=-1, keepdims=True)
    
    # Get weighted sum of V by attention_weights
    # shape: (n_heads, seq_len, head_dims)
    attention_output = np.matmul(attention_weights, V)
    
    # Combine head outputs
    # shape: (seq_len, n_dim)
    concat_output = self.combine_heads(attention_output)
    
    # Final output projection
    # shape: (seq_len, n_dim)
    self.output = np.dot(concat_output, self.wo)
    return self.output
  
  def backward(self, output_error, learning_rate):
    """
    Backward pass through attention layer.
    
    Algo:
			- We need the gradients for our 4 learnable matrices:
   
			- Wo error
						- O = Concat(head[1],...,head[n])Wo
						- dLoss/dWo = Concat(head[1],...,head[n]).t * dLoss/dOutput
						- dLoss/dConcat(head) = dLoss/dOutput * Wo.t
      
      - For our QKV matrices, we need to backpropagate through:
				- Value projection (weighted sum of attention scores)
				- QK interaction (score that determines attention weights)
				- Softmax of scores
      
      - V error
				- dLoss/dV = A.t * dLoss/dHead

				Attention weight gradient:
				- dLoss/dA = dLoss/dHead * V.t
    
				Scaled dot product gradient:
				- 
      
			- Q error
			- K error
    """
    
    # Gradient of output projection layer Wo
    wo_error = np.dot(self.output.transpose(0, 2, 1).reshape(-1, self.n_dim).T, output_error.reshape(-1, self.n_dim))
    concat_output_error = np.dot(output_error, self.wo.T).reshape(self.input.shape[0], -1, self.n_dim)
    
    # Split error between heads
    head_errors = self.split_heads(concat_output_error)
    
    # Gradient for attention weights / V
    attention_error = np.matmul(head_errors, self.wv.T)
    value_error = np.matmul(attention_error.transpose(1, 3, 2), self.wv.T)
    
    # Backprop QKV
    wq_error = np.dot(self.input.T, value_error)
    wk_error = np.dot(self.input.T, attention_error)
    wv_error = np.dot(self.input.T, head_errors.reshape(-1, self.n_dim))
    
    # Update weights
    self.wq -= learning_rate * wq_error
    self.wk -= learning_rate * wk_error
    self.wv -= learning_rate * wv_error
    self.wo -= learning_rate * wo_error
    
    input_error = np.dot(value_error, self.wq.T)
    return input_error 