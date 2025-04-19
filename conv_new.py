import numpy as np
from activation import leakyReLU, leakyReLU_derivative, sigmoid, sigmoid_derivative

class Convolutional:

    def __init__(self, name, image_shape, num_filters, stride=1, size=3, activation=None):
        self.name = name
        self.image_shape = image_shape # (width, height, channels)
        self.stride = stride
        self.size = size
        self.activation = activation
        self.filters = np.random.randn(num_filters, self.size, self.size, self.image_shape[2]) # initialize the filter

    def forward(self, image):
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
        
        return out
    
    def backward(self, din, learning_rate):
        batch_size, in_h, in_w, channels = self.last_input.shape
        num_f, f_size, _, _ = self.filters.shape

        dfilt = np.zeros_like(self.filters)
        dout = np.zeros_like(self.last_input)
        
        for b in range((batch_size)):
            for f in range((num_f)):
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

        self.filters -= learning_rate * dfilt
        return dout