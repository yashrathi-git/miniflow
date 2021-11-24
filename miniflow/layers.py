import numpy as np

from miniflow.core import Base


class LayerDense(Base):
    def __init__(self, n_previous, n_current):
        self.weights = np.random.randn(n_current, n_previous) * 0.01
        self.biases = np.zeros((n_current, 1))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(self.weights, inputs) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(dvalues, self.inputs.T)
        self.dbias = np.sum(dvalues, axis=1, keepdims=True)
        self.dinputs = np.dot(self.weights.T, dvalues)
