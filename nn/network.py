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

    def predict(self, input_data):
        n = input_data.shape[0]
        result = []
        
        for sample in range(n):
            output = input_data[sample]
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)
        
        return result
    
    def fit(self, x_train, y_train, epochs):
        if (self.loss == None) | (self.d_loss == None):
            raise AttributeError('Attributes loss or d_loss have not been set using the build() method.')
        
        n = x_train.shape[0]
        
        for epoch in range(epochs):
            # Set metric and error to 0
            metric = 0
            error = 0
            
            for sample in range(n):
                learning_rate = self.learning_rate_schedule()

                # Set output to input in case there is no layer
                output = x_train[sample]
                
                # Get output of all layers
                for layer in self.layers:
                    output = layer.forward(output)
                    
                # Calculate metric, loss and gradient for each sample
                metric += self.metric(y_train[sample], output)
                error += self.loss(y_train[sample], output)
                d_error = self.d_loss(y_train[sample], output)
                
                # Backpropagate gradient
                for layer in self.layers[::-1]:
                    d_error = layer.backward(d_error, learning_rate)
                               
            error /= n
            metric /= n
            
            self.history.append({'Epoch':epoch,
                                 'LR':learning_rate,
                                 'Loss':error,
                                 'Metric':metric})
            
            print(f"Epoch {epoch}/{epochs} Loss:{error} Metric:{metric}")