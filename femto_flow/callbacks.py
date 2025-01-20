import pickle

class SaveOnProgressCallback:
  def __init__(self, save_path):
    """Callback that takes a model as an input and saves it if the error has decreased."""
    self.min_error = 1e8
    self.save_count = 0
    self.save_path = save_path
  
  def __call__(self, model):
    if not hasattr(model, 'val_error'):
      raise AttributeError('Model has no validation loss. Make sure you are passing a model that has been trained with validation.')
    
    if model.val_error < self.min_error:
      self.min_error = model.val_error
      current_save_path = self.save_path + '/' + str(self.save_count) + ".pkl"
      with open(current_save_path, 'wb') as f:
        pickle.dump(model, f)
      self.save_count += 1
    else:
      pass

class PrintLRCallback:
  def __init__(self):
    pass

  def __call__(self, model):
    print(f"Learning rate: {model.learning_rate_schedule.lr:.3e}")