import numpy as np

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1-np.tanh(x)**2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    return 1 / (1 + np.exp(-x)) * (1 - (1 / (1 + np.exp(-x))))

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # numerically stable
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def d_softmax(x):
    s = softmax(x)
    return s * (1 - s)