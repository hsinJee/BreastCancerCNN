import numpy as np

class Dense:
    def __init__(self, name, input_size, output_size):
        self.name = name

        # Xavier initialization
        limit = np.sqrt(6 / (input_size + output_size))
        self.weights = np.random.uniform(-limit, limit, (input_size, output_size))
        self.biases = np.zeros(output_size)
        self.last_input = None
    
    def softmax(self, x):
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)
    
    def forward(self, input):
        self.last_input = input

        output = np.dot(input, self.weights) + self.biases
        self.last_output = output
        return self.softmax(output)
    
    def backward(self, din, learning_rate):
        batch_size = self.last_input.shape[0]

        dw = np.dot(self.last_input.T, din) / batch_size
        db = np.sum(din, axis=0) / batch_size

        self.weights -= learning_rate * dw
        self.biases -= learning_rate * db

        dout = np.dot(din, self.weights.T)

        return dout
    
    def get_weights(self):
        return np.reshape(self.weights, -1)
    
    # add property to save into best model file
    @property
    def params(self):
        return {
            'weights': self.weights,
            'biases': self.biases
        }
    
    @params.setter
    def params(self, new_params):
        self.weights = new_params['weights']
        self.biases = new_params['biases']