import numpy as np

epsilon = 1e-15

def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))
    
def d_mse(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.shape[0]
    
def bce(y_true, y_pred):
    y_pred = np.clip(y_pred, epsilon, 1-epsilon) # prevent division by 0
    return -np.mean((y_true * np.log(y_pred)) + ((1-y_true) * np.log((1-y_pred))))
    
def d_bce(y_true, y_pred):
    y_pred = np.clip(y_pred, epsilon, 1-epsilon) # prevent division by 0
    return -(y_true / y_pred - (1-y_true) / (1-y_pred))

def cce(y_true, y_pred):
    y_pred = np.clip(y_pred, epsilon, 1-epsilon) # prevent division by 0
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        
def d_cce(y_true, y_pred):
    y_pred = np.clip(y_pred, epsilon, 1-epsilon) # prevent division by 0
    return (y_pred - y_true) / y_true.shape[0]