class Flatten:
    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)
    
    def backward(self, dout, learning_rate):
        return dout.reshape(self.input_shape)