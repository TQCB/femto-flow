import pickle

class SaveOnProgressCallback:
  def __init__(self, save_path):
    """Callback that takes a model as an input and saves it if the error has decreased."""
    self.min_error = 99999
    self.save_count = 0
    self.save_path = save_path
  
  def __call__(self, model):
    if model.error < self.min_error:
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