import numpy as np

def batch(data, batch_size):
  '''
  Transform data from (samples, features) into (batch, samples, features)
  '''
  # Find out how many extra samples we have and remove them
  extra = data.shape[0] % batch_size
  data = data[:-extra,:]
  
  return data.reshape((-1, batch_size, data.shape[1]))