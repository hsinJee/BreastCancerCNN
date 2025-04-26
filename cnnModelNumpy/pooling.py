import numpy as np

class Pooling: # max pool
    def __init__(self, name, stride=2, size=2):
        self.name = name
        self.last_input = None
        self.stride = stride
        self.size = size

    def forward(self, image, training):
        self.last_input = image

        batch_size, in_h, in_w, channels = image.shape

        h = int((in_h - self.size) / self.stride) + 1 
        w = int((in_w - self.size) / self.stride) + 1

        downsampled = np.zeros((batch_size, h, w, channels))

        for b in range((batch_size)):
            for c in range(channels):
                out_y = 0
                for y in range(0, in_h - self.size + 1, self.stride):
                    out_x = 0
                    for x in range(0, in_w - self.size + 1, self.stride):
                        patch = image[b, y:y + self.size, x:x + self.size, c]
                        downsampled[b, out_y, out_x, c] = np.max(patch)
                        out_x += 1
                    out_y += 1

        return downsampled
    
    def backward(self, din, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        batch_size, in_h, in_w, channels = self.last_input.shape

        dout = np.zeros(self.last_input.shape)

        for b in range((batch_size)):
            for c in range(channels):
                out_y = 0
                for y in range(0, in_h - self.size + 1, self.stride):
                    out_x = 0
                    for x in range(0, in_w - self.size + 1, self.stride):
                        patch = self.last_input[b, y:y + self.size, x:x + self.size, c]
                        (max_x, max_y) = np.unravel_index(np.nanargmax(patch), patch.shape)
                        dout[b, y + max_x, x + max_y, c] += din[b, out_y, out_x, c]
                        out_x += 1
                    out_y += 1
        
        return dout
    
    def get_weights(self):                          # pooling layers have no weights
        return 0
                        