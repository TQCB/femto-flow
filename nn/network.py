import numpy as np

class Network:
    def __init__(self):
        self.layers = []
        self.history = []
        
        self.loss = None
        self.d_loss = None

    def add(self, layer):
        self.layers.append(layer)
    
    def build(self, loss, d_loss, metric, learning_rate_schedule):
        self.loss = loss
        self.d_loss = d_loss
        self.metric = metric
        self.learning_rate_schedule = learning_rate_schedule

    def predict(self, input_data, batched_output=True):
        # Number of batches
        batches = input_data.shape[0]
        result = []
        
        for batch in range(batches):
            output = input_data[batch]
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)
        
        # Instead of a list of batch results, we return a tensor stack
        # If there is only one result, just return the array of result
        result = np.stack(result) if len(result) > 1 else np.array(result)

        if batched_output:
            return result
        else:
            return result.reshape((-1, result.shape[2]))
    
    def fit(self, x_train, y_train, epochs, train_steps=None, val_steps=None, validation=True, x_val=None, y_val=None, callbacks=[], batch_print_steps=None):
        if validation & ((x_val is None) | (y_val is None)):
            raise ValueError('Validation data must be provided if you want to validate during fit')

        if (self.loss == None) | (self.d_loss == None):
            raise AttributeError('Attributes loss or d_loss have not been set using the build() method.')
        
        # Amount of batches
        train_batches = x_train.shape[0]
        if train_steps is None:
            train_steps = train_batches
        if train_steps > train_batches:
            raise ValueError("Cannot have more training steps than batches.")

        if validation:
            val_batches = x_val.shape[0]
            if val_steps is None:
                val_steps = val_batches
            if val_steps > val_batches:
                raise ValueError("Cannot have more validation steps than batches.")

        for epoch in range(epochs):
            # Set metric and error to 0
            metric = 0
            self.error = 0
            
            for batch in range(train_steps):
                # Randomly select a batch
                i = batch
                batch = np.random.randint(train_batches)

                if batch_print_steps is not None:
                    if i % batch_print_steps == 0:
                        print(f"Batch {i}/{train_steps} Loss:{self.error/(i+1):.3f} Metric:{metric/(i+1):.3f}", end="")
                        print("\r", end="")

                learning_rate = self.learning_rate_schedule()

                # Set output to input in case there is no layer
                output = x_train[batch]
                
                # Get output of all layers
                for layer in self.layers:
                    output = layer.forward(output)
                    
                # Calculate metric, loss and gradient for each batch
                metric += self.metric(y_train[batch], output)
                self.error += self.loss(y_train[batch], output)
                d_error = self.d_loss(y_train[batch], output)
                
                # Backpropagate gradient
                for layer in self.layers[::-1]:
                    d_error = layer.backward(d_error, learning_rate)
                    if d_error is not None:
                        d_error = np.clip(d_error, -1, 1)

            self.error /= train_steps
            metric /= train_steps
            
            self.history.append({'epoch':epoch,
                                 'lr':learning_rate,
                                 'loss':self.error,
                                 'metric':metric,})
            
            print(f"Epoch {epoch}/{epochs} Loss:{self.error:.3f} Metric:{metric:.3f}", end='')

            # If we use validation, we add val info the print and to the history
            if validation:
                val_error = 0
                val_metric = 0

                # Select val_steps random batches from val data
                sample_idx = np.random.randint(val_batches, size=val_steps)
                sample_x_val = x_val[sample_idx]
                sample_y_val = y_val[sample_idx]

                val_pred = self.predict(sample_x_val)
                for batch in range(val_steps):
                    val_error += self.loss(sample_y_val[batch], val_pred[batch])
                    val_metric += self.metric(sample_y_val[batch], val_pred[batch])

                val_error /= val_steps
                val_metric /= val_steps

                print(f" Val. Loss: {val_error:.3f} Val. Metric: {val_metric:.3f}", end='')
                self.history[len(self.history)-1] = {**self.history[len(self.history)-1], **{'val_loss' : val_error, 'val_metric': val_metric}}

            print()

            for callback in callbacks:
                callback(self)