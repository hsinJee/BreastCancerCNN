import numpy as np
from scipy.signal import correlate2d, convolve2d
from activation import leakyReLU, leakyReLU_derivative, sigmoid, sigmoid_derivative



class Convolutional:

    def __init__(self, name, image_shape, num_filters, stride=1, size=3, padding=0, activation=None):
        self.name = name
        self.image_shape = image_shape # (depth, height, width)
        self.stride = stride
        self.size = size
        in_channels, _, _ = image_shape
        self.filters = np.random.randn(num_filters, in_channels, self.size, self.size) * 0.1 # start with a 3x3 filter standard size
        self.activation = activation # e.g. relu
        self.last_input = None # cache for backwards pass
        self.last_output = None # cache for backwards pass
        self.padding = padding
        self.leakyReLU = np.vectorize(leakyReLU)
        self.leakyReLU_derivative = np.vectorize(leakyReLU_derivative)
        self.sigmoid = np.vectorize(sigmoid)
        self.sigmoid_derivative = np.vectorize(sigmoid_derivative)

    def forward(self, image): 
        # expected image shape (batch_size, depth, height, width)
        # note: batch size refers to size of each batch not total amount of batches
        assert image.shape[1:] == self.image_shape, \
            f"Input shape {image.shape} doesn't match layer expectation {self.image_shape} {self.name}"
        
        # apply padding if necessary
        if self.padding > 0:
            input = np.pad(image, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        else:
            input = image
            
        batch_size, channels, input_height, input_width = input.shape
        num_filters = self.filters.shape[0]

        self.last_input = input # cache for backwards pass

        # height and width should have the same value
        output_height = int((input_height - self.size + (2 * self.padding)) // self.stride) + 1 # output height calculations
        output_width = int((input_width - self.size + (2 * self.padding)) // self.stride) + 1 # output width calculations

        out = np.zeros((batch_size, num_filters, output_height, output_width)) # create matrix to hold values of convolution
        

        for b in range((batch_size)):
            for f in range((num_filters)):
                tmp_y = out_y = 0
                while tmp_y + self.size <= input_height:
                    tmp_x = out_x = 0
                    while tmp_x + self.size <= input_width:
                        patch = input[b , :, tmp_y:tmp_y + self.size, tmp_x:tmp_x + self.size]
                        out[b, f, out_y, out_x] += np.sum(self.filters[f] * patch)
                        tmp_x += self.stride
                        out_x += 1
                    tmp_y += self.stride
                    out_y += 1
        if self.activation == 'relu': # apply ReLU activation function
            self.leakyReLU(out)
        elif self.activation == 'sigmoid':
            self.sigmoid(out)

        self.last_output = out
        return out
    
    def backward(self, din, learn_rate=0.005):
        batch_size, channels, input_height, input_width  = self.last_input.shape # input dimensions
        num_filters = self.filters.shape[0]

        if self.activation == 'relu': 
            self.leakyReLU_derivative(din)
        elif self.activation == 'sigmoid':
            self.sigmoid_derivative(din)
        
        # initialize the loss gradient of the input and filter
        dout = np.zeros(self.last_input.shape) 
        dfilt = np.zeros(self.filters.shape) 


        for b in range(batch_size):
            for f in range(num_filters):
                tmp_y = out_y = 0
                while tmp_y + self.size <= input_height:
                    tmp_x = out_x = 0
                    while tmp_x + self.size <= input_width:
                        patch = self.last_input[b, :, tmp_y:tmp_y + self.size, tmp_x:tmp_x + self.size]
                        dfilt[f] += din[b, f, out_y, out_x] * patch
                        dout[b, :, tmp_y:tmp_y + self.size, tmp_x:tmp_x + self.size] += din[b, f, out_y, out_x]
                        tmp_x += self.stride
                        out_x += 1
                    tmp_y += self.stride
                    out_y += 1        

        # remove the paddings from d_input if added
        if self.padding > 0:
            dout = dout[:, :, self.padding:-self.padding, self.padding:-self.padding]  

        # update the filters
        self.filters -= learn_rate * dfilt

        return dout
    
    def get_weights(self):
        return np.reshape(self.filters, -1)