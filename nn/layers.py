import numpy as np

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
    def __init__(self, activation, d_activation):
        self.activation = activation
        self.d_activation = d_activation

    # compute h(X) = Y
    def forward_propagation(self, input):
        self.input = input
        self.output = self.activation(self.input)
        return self.output
    
    # compute dE/dX for dE/dY 
    def backward_propagation(self, output_error, learning_rate):
        return self.d_activation(self.input) * output_error
    
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