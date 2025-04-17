import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def leakyReLU(x, alpha=0.001):
    return np.where(x >= 0, x, alpha * x)

def leakyReLU_derivative(x, alpha=0.01):
    return np.where(x >= 0, 1, alpha)

