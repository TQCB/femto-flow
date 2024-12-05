import numpy as np

def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
    
def binary_accuracy(y_true, y_pred, threshold=0.5):
    y_pred = (y_pred >= threshold) + 0
    return np.sum((y_pred == y_true)) / y_true.shape[0]

def categorical_accuracy(y_true, y_pred):    
    y_true = np.reshape(y_true, (1, -1))
    y_pred = np.reshape(y_pred, (1, -1))
    
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))