import numpy as np

class Dense:
    def __init__(self, name, input_size, output_size, useSoftmax):
        self.name = name
        limit = np.sqrt(6 / (input_size + output_size))
        self.weights = np.random.uniform(-limit, limit, (input_size, output_size))
        self.biases = np.zeros(output_size)
        self.last_input = None
        self.useSoftmax = useSoftmax

        # Adam variables initialization
        self.m_w = np.zeros_like(self.weights)
        self.v_w = np.zeros_like(self.weights)
        self.m_b = np.zeros_like(self.biases)
        self.v_b = np.zeros_like(self.biases)
        self.t = 0
    
    def softmax(self, x):
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)
    
    def forward(self, input, training):
        self.last_input = input

        output = np.dot(input, self.weights) + self.biases
        self.last_output = output

        if self.useSoftmax:
            output = self.softmax(output)

        return output
    
    def backward(self, din, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        batch_size = self.last_input.shape[0]
        dw = np.dot(self.last_input.T, din) / batch_size
        db = np.sum(din, axis=0) / batch_size

        self.t += 1

        # Update weights using Adam
        self.m_w = beta1 * self.m_w + (1 - beta1) * dw
        self.v_w = beta2 * self.v_w + (1 - beta2) * (dw ** 2)
        m_w_hat = self.m_w / (1 - beta1 ** self.t)
        v_w_hat = self.v_w / (1 - beta2 ** self.t)
        self.weights -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)

        # Update biases using Adam
        self.m_b = beta1 * self.m_b + (1 - beta1) * db
        self.v_b = beta2 * self.v_b + (1 - beta2) * (db ** 2)
        m_b_hat = self.m_b / (1 - beta1 ** self.t)
        v_b_hat = self.v_b / (1 - beta2 ** self.t)
        self.biases -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

        dout = np.dot(din, self.weights.T)
        return dout
    
    def get_weights(self):
        return np.reshape(self.weights, -1)
    
    def set_weights(self, x):
        self.weights = x.reshape(self.weights.shape)
    
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