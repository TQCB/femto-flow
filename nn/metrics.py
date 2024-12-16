import numpy as np

def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
    
def binary_accuracy(y_true, y_pred, threshold=0.5):
    y_pred = (y_pred >= threshold) + 0
    return np.sum((y_pred == y_true)) / y_true.shape[0]

def categorical_accuracy(y_true, y_pred):
    batch_size = y_true.shape[0]
    accuracy = []
    for sample in range(batch_size):
        accuracy.append(np.argmax(y_true[sample]) == np.argmax(y_pred[sample]))
    return np.mean(accuracy)