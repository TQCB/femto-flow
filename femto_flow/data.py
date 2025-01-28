import numpy as np

def batch(data, batch_size, fill=False):
  '''
  Transform data from (sample, feature) into (batch, sample, feature)
  '''
  # Find out how many extra samples we have and remove them
  extra = data.shape[0] % batch_size
  if extra != 0:
    data = data[:-extra,:]
  
  return data.reshape((-1, batch_size, data.shape[1]))

def load_data(path, encoding='utf-8'):
  with open(path, 'r', encoding=encoding) as f:
    data = f.read()
  return data

def targets_from_sequence(sequence, context_size):
  """
  Constructs n times X, y training data from a sequence. X being our
  context_size input,  y the target, and n the amount of targets created from a
  sequence (should be equal to len(sequence) - 1, since every word can be a
  target except the first)

  X is context_size words used to predict context_size + 1 word y

  Args:
    sequence (array): array of ints, vectorised tokens
    context_size (int): length of X for each X, y pair returned
  
  Returns:
    list[(X,y)]: n long list containing tuples of X, y pairs
  """
  target_amount = len(sequence) - 1

  # Ensure input in an np.ndarray
  sequence = np.array(sequence)

  # Result array is a matrix for (n, (X,y))
  # n = target_amount, X = context_size, y = 1
  result = np.zeros((target_amount, context_size+1))


  for i in range(target_amount):
    # Get index of context start, unless its negative
    idx = i - context_size + 1
    context_start = max(0, idx)

    # Get sequence and target, ensuring they are arrays
    context = sequence[context_start:i+1]
    target = np.array(sequence[i+1], ndmin=1)

    # If our context is shorter than context size, we will pad context
    if context_start == 0:
      context = np.pad(context, (0, np.abs(idx)), mode='constant')

    result[i,:] = np.concatenate([context, target])
    
  return result

def targets_from_sequences(sequences, context_size):
  """Calls targets_from_sequences() on a list of sequences"""
  result = []
  for sequence in sequences:
    result.append(targets_from_sequence(sequence, context_size))
  return np.vstack(result)

def one_hot_encode(x, n_classes=None, dtype='int'):
  if n_classes is None:
    n_classes = np.max(x) # max class number found in array
  return np.eye(n_classes, dtype=dtype)[x]

def train_test_split(data, ratio):
    # Split x, y
    x, y = data[:,:-1], data[:,-1]

    # Split train, test
    choice = np.random.choice(data.shape[0], size=int(ratio*data.shape[0]), replace=False)
    idx = np.zeros(data.shape[0], dtype=bool)
    idx[choice] = True

    x_train, x_val = x[~idx], x[idx]
    y_train, y_val = y[~idx], y[idx]

    return x_train, x_val, y_train, y_val