class Flatten:
    def forward(self, x, training):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)
    
    def backward(self, dout, learning_rate,  beta1=0.9, beta2=0.999, epsilon=1e-8):
        return dout.reshape(self.input_shape)
    
    def get_weights(self):
        return 0