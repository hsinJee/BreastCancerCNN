import numpy as np
# from activation import leakyReLU, leakyReLU_derivative, sigmoid, sigmoid_derivative

class Convolutional:

    def __init__(self, name, image_shape, num_filters, stride=1, size=3, activation=None):
        self.name = name
        self.image_shape = image_shape
        self.stride = stride
        self.size = size
        self.activation = activation
        self.filters = np.random.randn(num_filters, size, size, image_shape[2]) * np.sqrt(2. / (size * size * image_shape[2]))

        # Adam variables
        self.m = np.zeros_like(self.filters)
        self.v = np.zeros_like(self.filters)
        self.t = 0

    def forward(self, image, training):
        batch_size, in_h, in_w, channels = image.shape
        num_f, f_size, _, _ = self.filters.shape
        

        self.last_input = image # cache for backward pass

        out_h = int((in_h - f_size) // self.stride) + 1
        out_w = int((in_w - f_size) // self.stride) + 1

        out = np.zeros((batch_size, out_h, out_w, num_f))

        for b in range((batch_size)):
            for f in range((num_f)):
                out_y = 0
                for y in range(0, in_h - self.size + 1, self.stride):
                    out_x = 0
                    for x in range(0, in_w - f_size + 1, self.stride):
                        patch = image[b,  y:y + f_size, x:x + f_size, :]
                        out[b, out_y, out_x, f] += np.sum(patch * self.filters[f])
                        out_x += 1
                    out_y += 1

        self.last_output = out.copy()

        if self.activation == 'relu':
            out = np.maximum(0, out)  # ReLU
        elif self.activation == 'sigmoid':
            out = 1 / (1 + np.exp(-out))  # Sigmoid
        elif self.activation == 'tanh':
            out = np.tanh(out)  # Tanh
        else:
            out = out
        return out
    
    def backward(self, din, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        batch_size, in_h, in_w, channels = self.last_input.shape
        num_f, f_size, _, _ = self.filters.shape

        if self.activation == 'relu':
            dout = din * (self.last_output > 0)
        elif self.activation == 'sigmoid':
            sigmoid_out = 1 / (1 + np.exp(-self.last_output))
            dout = din * sigmoid_out * (1 - sigmoid_out)
        elif self.activation == 'tanh':
            tanh_out = np.tanh(self.last_output)
            dout = din * (1 - tanh_out ** 2)
        else:
            dout = din  # No activation

        dfilt = np.zeros_like(self.filters)
        dout = np.zeros_like(self.last_input)

        for b in range(batch_size):
            for f in range(num_f):
                out_y = 0
                for y in range(0, in_h - self.size + 1, self.stride):
                    out_x = 0
                    for x in range(0, in_w - f_size + 1, self.stride):
                        logits = din[b, out_y, out_x, f]
                        patch = self.last_input[b, y:y+f_size, x:x+f_size, :]

                        dfilt[f] += patch * logits
                        dout[b, y:y+f_size, x:x+f_size, :] += self.filters[f] * logits

                        out_x += 1
                    out_y += 1

        # Adam update
        self.t += 1
        self.m = beta1 * self.m + (1 - beta1) * dfilt
        self.v = beta2 * self.v + (1 - beta2) * (dfilt ** 2)

        m_hat = self.m / (1 - beta1 ** self.t)
        v_hat = self.v / (1 - beta2 ** self.t)

        self.filters -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        return dout
    
    def get_weights(self):
        return np.reshape(self.filters, -1)
    
    def set_weights(self, x):
        self.filters = x.reshape(self.filters.shape)
    
    @property
    def params(self):
        return {'filters': self.filters}
    
    @params.setter
    def params(self, new_params):
        self.filters = new_params['filters']