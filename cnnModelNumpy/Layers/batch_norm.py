import numpy as np

class BatchNormalization:
    def __init__(self, name, epsilon=1e-5, momentum=0.9):
        self.name = name
        self.epsilon = epsilon
        self.momentum = momentum
        self.gamma = None
        self.beta = None
        self.running_mean = None
        self.running_var = None

        # Adam variables
        self.m_gamma = None
        self.v_gamma = None
        self.m_beta = None
        self.v_beta = None
        self.t = 0

    def build(self, input_shape):
        channels = input_shape[-1]
        self.gamma = np.ones((1, 1, 1, channels))
        self.beta = np.zeros((1, 1, 1, channels))
        self.running_mean = np.zeros((1, 1, 1, channels))
        self.running_var = np.ones((1, 1, 1, channels))

        # initialize Adam variables
        self.m_gamma = np.zeros_like(self.gamma)
        self.v_gamma = np.zeros_like(self.gamma)
        self.m_beta = np.zeros_like(self.beta)
        self.v_beta = np.zeros_like(self.beta)

    def forward(self, x, training=True):
        # x is input, x_norm is the normalized input
        if self.gamma is None:
            self.build(x.shape)
        
        # store for backward pass
        self.x = x
        
        if training:
            self.mean = np.mean(x, axis=(0, 1, 2), keepdims=True)
            self.var = np.var(x, axis=(0, 1, 2), keepdims=True)

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var

            self.x_norm = (x - self.mean) / np.sqrt(self.var + self.epsilon)
        else:
            self.x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        
        out = self.gamma * self.x_norm + self.beta
        return out
    
    def backward(self, dout, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        assert hasattr(self, "x"), "Forward must be called before backward!"
        N, H, W, C = dout.shape
        N_total = N * H * W

        dgamma = np.sum(dout * self.x_norm, axis=(0, 1, 2), keepdims=True)
        dbeta = np.sum(dout, axis=(0, 1, 2), keepdims=True)

        dx_norm = dout * self.gamma
        std_inv = 1. / np.sqrt(self.var + self.epsilon)

        dx = (1. /  N_total) * std_inv * (
            N_total * dx_norm - np.sum(dx_norm, axis=(0, 1, 2), keepdims=True)
            - self.x_norm * np.sum(dx_norm * self.x_norm, axis=(0, 1, 2), keepdims=True)
        )

        # Adam update
        self.t += 1

        # update gamma
        self.m_gamma = beta1 * self.m_gamma + (1 - beta1) * dgamma
        self.v_gamma = beta2 * self.v_gamma + (1 - beta2) * (dgamma ** 2)
        m_gamma_hat = self.m_gamma / (1 - beta1 ** self.t)
        v_gamma_hat = self.v_gamma / (1 - beta2 ** self.t)
        self.gamma -= learning_rate * m_gamma_hat / (np.sqrt(v_gamma_hat) + epsilon)

        # update beta
        self.m_beta = beta1 * self.m_beta + (1 - beta1) * dbeta
        self.v_beta = beta2 * self.v_beta + (1 - beta2) * (dbeta ** 2)
        m_beta_hat = self.m_beta / (1 - beta1 ** self.t)
        v_beta_hat = self.v_beta / (1 - beta2 ** self.t)
        self.beta -= learning_rate * m_beta_hat / (np.sqrt(v_beta_hat) + epsilon)

        return dx
    
    
    def get_weights(self):
        return np.concatenate([
            self.gamma.flatten(),
            self.beta.flatten()
        ])

    @property
    def params(self):
        return {'gamma': self.gamma, 'beta': self.beta, 'running_mean': self.running_mean, 'running_var': self.running_var}

    @params.setter
    def params(self, new_params):
        self.gamma = new_params['gamma']
        self.beta = new_params['beta']
        self.running_mean = new_params.get('running_mean', np.zeros_like(self.gamma))
        self.running_var = new_params.get('running_var', np.ones_like(self.gamma))