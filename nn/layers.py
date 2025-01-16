import numpy as np

################################################################################

# Module containing definitions of different layers for my neural networks.
# Layers at a minimum define forward and backward pass methods. These 
# respectively apply the layer's calculations to an input, passing the result to
# the next layer and propagating the error to the previous layer.

################################################################################

class Layer():
  def __init__(self):
    pass

  def initialize_optimizer(self, optimizer):
    pass

  def get_param_count(self, weights):
    return np.sum([weight.size for weight in weights])

class MetaLayer(Layer):
  def __init__(self, layers, clip=0):
    '''Layer that takes multiple layers and forward/backward passes through all of them sequentially'''
    self.layers = layers
    self.clip = clip

    self.param_count = np.sum([layer.param_count for layer in self.layers])

  def initialize_optimizer(self, optimizer):
    for layer in self.layers:
      layer.initialize_optimizer(optimizer)

  def forward(self, input):
    output = input
    for layer in self.layers:
      output = layer.forward(input)
    
    return output

  def backward(self, output_error, learning_rate):
    input_error = output_error
    for layer in self.layers[::-1]:
      input_error = layer.backward(input_error, learning_rate)
      if self.clip > 0:
        input_error = np.clip(input_error, -self.clip, self.clip)

    return input_error

class Activation(Layer):
  def __init__(self, activation):
    self.activation = activation()
    self.param_count = 0

  # compute h(X) = Y
  def forward(self, input):
    self.input = input
    self.output = self.activation.forward(self.input)
    return self.output
    
  # compute dE/dX for dE/dY 
  def backward(self, output_error, learning_rate):
    return self.activation.backward(self.input) * output_error
    
class Dense1D(Layer):
  def __init__(self, input_dim, output_dim):
    # Xavier/Glorot uniform initialization
    limit = np.sqrt(6 / (input_dim + output_dim))

    self.weights = np.random.uniform(-limit, limit, size=(input_dim, output_dim))
    self.bias = np.zeros((1,output_dim))

    self.param_count = self.get_param_count([self.weights, self.bias])

  def initialize_optimizer(self, optimizer):
    self.opt_weights = optimizer(self.weights)
    self.opt_bias = optimizer(self.bias)

  def forward(self, input):
    self.input = input # shape: (batch, feature)
    self.output = np.dot(self.input, self.weights) + self.bias # shape: (batch, neuron)
    return self.output
    
  def backward(self, output_error, learning_rate):
    batch_size = output_error.shape[0]

    input_error = np.dot(output_error, self.weights.T)
    weights_error = np.dot(self.input.T, output_error)
    weights_error /= batch_size
        
    # Calculate the gradient of the bias by summing over the batch dimension (axis=0)
    bias_error = np.sum(output_error, axis=0, keepdims=True) # shape: 
    bias_error /= batch_size
        
    self.weights -= learning_rate * weights_error
    self.bias -= learning_rate * bias_error
    return input_error
  
class Dense2D(Layer):
  def __init__(self, input_dim, output_dim):
    # Xavier/Glorot uniform initialization
    limit = np.sqrt(6 / (input_dim + output_dim))
    self.weights = np.random.uniform(-limit, limit, size=(input_dim, output_dim))
    self.bias = np.zeros((1, output_dim))

    self.param_count = self.get_param_count([self.weights, self.bias])

  def initialize_optimizer(self, optimizer):
    self.opt_weights = optimizer(self.weights)
    self.opt_bias = optimizer(self.bias)

  def forward(self, input):
    self.input = input # shape: (batch, seq_len, feature)
    self.output = np.dot(self.input, self.weights) + self.bias # shape: (batch, neuron)
    return self.output
    
  def backward(self, output_error, learning_rate):
    input_error = np.dot(output_error, self.weights.T)
    weights_error = np.matmul(self.input.transpose(0, 2, 1), output_error)
    weights_error = np.mean(weights_error, axis=0)
        
    # Calculate the gradient of the bias by summing over the batch dimension (axis=0)
    bias_error = np.mean(output_error, axis=0) # shape: 3, 8
    bias_error = np.sum(bias_error, axis=0, keepdims=True) # shape: 1, 8

    self.weights = self.opt_weights.apply_gradients(self.weights, weights_error, learning_rate)
    self.bias = self.opt_bias.apply_gradients(self.bias, bias_error, learning_rate)

    return input_error
    
class MultiHeadSelfAttention(Layer):
  """
  Multi Head Self Attention for Dense networks.
    
  Args:
		input dim: Dimension of input (most likely the sequence len)
		n_dim: Total dimensions for Q, K, V matrices
		n_heads: Quantity of attention heads
  """
    
  def __init__(self, input_dim, n_dim, n_heads, residual=True, causal=True, return_sequences=True):
    self.n_heads = n_heads
    self.n_dim = n_dim
    self.head_dim = n_dim // n_heads # dims for qkv per head
    self.residual = residual
    self.causal = causal
    self.return_sequences = return_sequences
    
    # Xavier/Glorot uniform initialization
    limit = np.sqrt(6 / (input_dim + n_dim))

    # Initialize weights for Q, K, V
    self.wq = np.random.uniform(-limit, limit, size=(input_dim, n_dim))
    self.wk = np.random.uniform(-limit, limit, size=(input_dim, n_dim))
    self.wv = np.random.uniform(-limit, limit, size=(input_dim, n_dim))
    
    # Output projection weights
    self.wo = np.random.rand(self.n_dim, self.n_dim) - 0.5

    self.param_count = self.get_param_count([self.wq, self.wk, self.wv, self.wo])
    
  def initialize_optimizer(self, optimizer):
    self.opt_wq = optimizer(self.wq)
    self.opt_wk = optimizer(self.wk)
    self.opt_wv = optimizer(self.wv)
    self.opt_wo = optimizer(self.wo)

  def split_heads(self, x):
    """
    Utility function to split global Q, K, V

    (batch, seq_len, n_dim) -> (batch, n_heads, seq_len, head_dim)
    """
    batches = x.shape[0]
    seq_len = x.shape[1]
    return x.reshape(batches, self.n_heads, seq_len, self.head_dim)
    
  def combine_heads(self, x):
    """
    Utility function to combine head Q, K, V
    
    (batch, n_heads, seq_len, head_dim) -> (batch, seq_len, n_dim)
    """
    batches = x.shape[0]
    seq_len = x.shape[2]
    return x.reshape(batches, seq_len, self.n_dim)
  
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
    self.Q = np.dot(self.input, self.wq)
    self.K = np.dot(self.input, self.wk)
    self.V = np.dot(self.input, self.wv)
    
    # Split Q, K, V into heads
    # shape: (n_heads, seq_len, head_dim)
    self.Q = self.split_heads(self.Q)
    self.K = self.split_heads(self.K)
    self.V = self.split_heads(self.V)
    
    # Compute Luong attention (scaled dot product)
    # Divide by sqrt(d) for more stable gradient flow
    # Normalizes the magnitude of score with respect to d
    # shape: (n_heads, seq_len, seq_len)
    # score of each Q to each K, each element of sequence to every other
    scores = np.matmul(self.Q, self.K.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
    
    # Softmax scores
    # shape: (n_heads, seq_len, seq_len)
    self.attention_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    self.attention_weights /= np.sum(self.attention_weights, axis=-1, keepdims=True)
    
    # Get weighted sum of V by attention_weights
    # shape: (n_heads, seq_len, head_dims)
    self.V = np.clip(self.V, -1, 1) # lots of overflows, clipping V helps enormously (DEBUG)
    attention_output = np.matmul(self.attention_weights, self.V)
    
    # Combine head outputs
    # shape: (seq_len, n_dim)
    concat_output = self.combine_heads(attention_output)
    
    # Final output projection
    # shape: (seq_len, n_dim)
    output = np.dot(concat_output, self.wo)

    try:
      assert self.input.shape == output.shape
    except AssertionError:
      raise ValueError("Attention input and output have different dimensions")

    # Add residual
    if self.residual:
      output += self.input

    # If we don't want to return sequences, only return last element in sequence
    if not self.return_sequences:
      output = output[:,-1,:]
    
    return output
  
  def backward(self, output_error, learning_rate):
    """
    Backward pass through attention layer.
    """
    
    if not self.return_sequences:
      # Expand output error to be same size as layer output, even if we only returned a sequence
      expanded_output_error = np.zeros_like(self.input)
      expanded_output_error[:, -1, :] = output_error
      output_error = expanded_output_error

    # Gradient of output projection layer Wo
    # shape: (batch, n_dim, n_dim)
    d_wo = np.matmul(self.combine_heads(np.matmul(self.attention_weights, self.V)).transpose(0, 2, 1), output_error)
    # shape: (batch, seq_len, n_dim)
    combined_output_error = np.dot(output_error, self.wo.T)

    # Split error between heads
    # shape: (batch, n_heads, seq_len, head_dim)
    output_error_heads = self.split_heads(combined_output_error)

    # Backprop through weighted sum
    self.V = np.clip(self.V, -1, 1) # lots of overflows, clipping V helps enormously (DEBUG)
    d_attention_weights = np.matmul(output_error_heads, self.V.transpose(0, 1, 3, 2))

    d_V = np.matmul(self.attention_weights.transpose(0, 1, 3, 2), output_error_heads)

    # Backprop through softmax
    d_scores = self.attention_weights * (d_attention_weights - np.sum(self.attention_weights * d_attention_weights, axis=-1, keepdims=True))
    
    # Backprop through scaling
    d_scores /= np.sqrt(self.head_dim)

    # Backprop through QK^T
    d_Q = np.matmul(d_scores, self.K)
    d_K = np.matmul(d_scores.transpose(0, 1, 3, 2), self.Q)

    # Combine gradients for each head
    d_Q_combined = self.combine_heads(d_Q)
    d_K_combined = self.combine_heads(d_K)
    d_V_combined = self.combine_heads(d_V)
    
    # Backprop through Q, K, V projections
    d_wq = np.matmul(self.input.transpose(0, 2, 1), d_Q_combined)
    d_wk = np.matmul(self.input.transpose(0, 2, 1), d_K_combined)
    d_wv = np.matmul(self.input.transpose(0, 2, 1), d_V_combined)

    # Input error
    d_input = np.dot(d_Q_combined, self.wq.T) + np.dot(d_K_combined, self.wk.T) + np.dot(d_V_combined, self.wv.T)

    # Add error from residual connection
    if self.residual:
      d_input += output_error

    # Update weights
    # self.wq -= learning_rate * d_wq.mean(axis=0)
    # self.wk -= learning_rate * d_wk.mean(axis=0)
    # self.wv -= learning_rate * d_wv.mean(axis=0)
    # self.wo -= learning_rate * d_wo.mean(axis=0)

    self.wq = self.opt_wq.apply_gradients(self.wq, d_wq.mean(axis=0), learning_rate)
    self.wk = self.opt_wk.apply_gradients(self.wk, d_wk.mean(axis=0), learning_rate)
    self.wv = self.opt_wv.apply_gradients(self.wv, d_wv.mean(axis=0), learning_rate)
    self.wo = self.opt_wo.apply_gradients(self.wo, d_wo.mean(axis=0), learning_rate)

    return d_input
    
class Embedding(Layer):
  def __init__(self, input_dim, output_dim, vocab_size):
    # Xavier/Glorot uniform initialization
    limit = np.sqrt(6 / (vocab_size + output_dim))

    # vocab_size rows, output_dim embedding dimensions
    self.global_embedding_weights = np.random.uniform(-limit, limit, size=(vocab_size, output_dim))
    self.vocab_size = vocab_size
    self.output_dim = output_dim

    self.param_count = self.get_param_count([self.global_embedding_weights])

  def initialize_optimizer(self, optimizer):
    self.opt_weights = optimizer(self.global_embedding_weights)

  def forward(self, input):
    # Input is a sequence of ints representing tokens
    # We get a local embedding for those token
    # shape: (seq_len, output_dim)
    self.input = input # Save the input sequence for backprop
    self.local_embedding_weights = self.global_embedding_weights[input]
    return self.local_embedding_weights

  def backward(self, output_error, learning_rate):
    # Create an array to accumulate gradients for each word in the vocabulary
    d_global_embedding_weights = np.zeros_like(self.global_embedding_weights)

    # Iterate through the input sequence and corresponding output error
    for i, word_index in enumerate(self.input):
        # Add the output error (gradient) for the current word to the accumulated gradients
        d_global_embedding_weights[word_index] += output_error[i]
    
    # Update global embedding weights using the accumulated gradients and learning rate
    self.global_embedding_weights -= learning_rate * d_global_embedding_weights

    # In an embedding layer, there's no gradient to pass back to the previous layer
    # (The input was just a sequence of integer indices)
    return None
  
class PositionalEmbedding(Layer):
  def __init__(self, seq_len, output_dim, vocab_size):
    # vocab_size rows, output_dim embedding dimensions
    self.global_embedding_weights = np.random.rand(vocab_size, output_dim)
    self.vocab_size = vocab_size
    self.seq_len = seq_len
    self.output_dim = output_dim

    self.positional_encoding = self.get_positional_encoding(self.output_dim)

    self.param_count = self.get_param_count([self.global_embedding_weights])

  def initialize_optimizer(self, optimizer):
    self.opt_weights = optimizer(self.global_embedding_weights)

  def get_positional_encoding(self, dim, n=10000):
    enc = np.empty([self.seq_len, dim])

    # Use positional encoding from "Attention is all you Need"
    for pos in range(self.seq_len):
      for i in range(int(dim/2)):
        enc[pos, 2*i] = np.sin(pos / (n ** (2*1/dim)))
        enc[pos, 2*i+1] = np.cos(pos / (n ** (2*1/dim)))

    return enc

  def forward(self, input):
    # Input is a sequence of ints representing tokens
    # We get a local embedding for those token
    # shape: (seq_len, output_dim)
    self.input = input # Save the input sequence for backprop
    self.batch_size = self.input.shape[0]

    self.local_embedding_weights = self.global_embedding_weights[input]
    return self.local_embedding_weights + self.positional_encoding

  def backward(self, output_error, learning_rate):
    # Create an array to accumulate gradients for each word in the vocabulary
    d_global_embedding_weights = np.zeros_like(self.global_embedding_weights)

    # Iterate through the input sequence and corresponding output error
    for i, word_index in enumerate(self.input):
        # Add the output error (gradient) for the current word to the accumulated gradients
        d_global_embedding_weights[word_index] += output_error[i]
    
    # Update global embedding weights using the accumulated gradients and learning rate
    self.global_embedding_weights -= learning_rate * d_global_embedding_weights

    # In an embedding layer, there's no gradient to pass back to the previous layer
    # (The input was just a sequence of integer indices)
    return None

class LayerNormalisation(Layer):
  def __init__(self, dim, axis=-1, eps=1e-4):
    self.dim = dim
    self.axis = axis

    self.gamma = np.ones(dim)
    self.beta = np.zeros(dim)
    self.eps = eps

    self.param_count = self.get_param_count([self.gamma, self.beta])

  def initialize_optimizer(self, optimizer):
    self.opt_gamma = optimizer(self.gamma)
    self.opt_beta = optimizer(self.beta)

  def forward(self, input):
    self.input = input
    self.mu = np.mean(input, axis=self.axis, keepdims=True)
    self.var = np.var(input, axis=self.axis, keepdims=True)

    self.output = (input - self.mu) / np.sqrt(self.var + self.eps)
    self.output = self.gamma * self.output + self.beta

    return self.output

  def backward(self, output_error, learning_rate):
    # Gradients w.r.t. gamma and beta
    self.gamma_error = np.sum(output_error * self.output, axis=(0,1))
    self.beta_error = np.sum(output_error, axis=(0,1))

    # Gradient w.r.t. output
    output_error *= self.gamma

    # Gradient w.r.t. variance
    var_error = output_error * (self.input - self.mu) * (-0.5) * (self.var + self.eps) ** (-1.5)
    var_error = np.sum(var_error, axis=self.axis, keepdims=True)

    # Gradient w.r.t. mu
    mu_error = output_error * (-1) / np.sqrt(self.var + self.eps)
    mu_error = np.sum(mu_error, axis=self.axis, keepdims=True)
    mu_error += var_error * np.mean(-2 * (self.input - self.mu), axis=self.axis, keepdims=True)
    
    # Gradient w.r.t. input
    input_error = (output_error / np.sqrt(self.var + self.eps)) + \
                  (var_error * 2 * (self.input - self.mu) / self.dim) + \
                  (mu_error / self.dim)
    
    # Update parameters
    self.gamma -= learning_rate * self.gamma_error
    self.beta -= learning_rate * self.beta_error
    
    # Clipping these helps a lot with stability
    clip_limit = 1e3
    self.gamma = np.clip(self.gamma, -clip_limit, clip_limit)
    self.beta = np.clip(self.beta, -clip_limit, clip_limit)

    return input_error
  
  class Dropout(Layer):
    def __init__(self, dropout_rate):
      self.dropout_rate = dropout_rate
      
    def forward(self, input):
      # Randomly generate a list of indices to drop
      # len(drop_idx) = dropout_rate * input.size
      self.drop_idx = int(dropout_rate * input.size)
      
    def backward(self, output_error, learning_rate):