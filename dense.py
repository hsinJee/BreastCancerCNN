import numpy as np
def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)  # More stable if axis=1 (batch-wise)
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=1, keepdims=True)

class Dense:
    def __init__(self, name, nodes, num_classes):
        self.name = name
        self.weights = np.random.randn(nodes, num_classes) * 0.1
        self.biases =  np.zeros(num_classes)
        self.last_input_shape = None
        self.last_input = None
        self.last_output = None
    
    def forward(self, input):
        self.last_input_shape = input.shape

        if input.ndim > 2:
            input = input.reshape(input.shape[0], -1)

        output = np.dot(input, self.weights) + self.biases

        self.last_input = input
        self.last_output = output

        return softmax(output)
    
    def backward(self, din, learning_rate=0.005):
        batch_size = self.last_input.shape[0]
        # Gradients
        dw = np.dot(self.last_input.T, din) / batch_size        
        db = np.sum(din, axis=0) / batch_size                                          

        # Update weights and biases
        self.weights -= learning_rate * dw
        self.biases -= learning_rate * db

        # Gradient to pass backward
        dout = np.dot(din, self.weights.T)                  
        return dout.reshape(self.last_input_shape)

    def get_weights(self):
        return np.reshape(self.weights, -1)
    
    
    