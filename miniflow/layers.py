import numpy as np

from miniflow.core import Base


class LayerDense(Base):
    def __init__(
        self,
        n_previous,
        n_current,
        weight_regularization_l1=0,
        weight_regularization_l2=0,
        bias_regularization_l1=0,
        bias_regularization_l2=0,
    ):
        # 2 / n[l-1] works well with reLU which is most commonly used
        self.weights = np.random.randn(n_current, n_previous) * np.sqrt(2 / n_previous)
        self.biases = np.zeros((n_current, 1))
        self.weight_regularization_l1 = weight_regularization_l1
        self.weight_regularization_l2 = weight_regularization_l2
        self.bias_regularization_l1 = bias_regularization_l1
        self.bias_regularization_l2 = bias_regularization_l2

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(self.weights, inputs) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(dvalues, self.inputs.T)
        self.dbias = np.sum(dvalues, axis=1, keepdims=True)

        # L1 regularization
        if self.weight_regularization_l1:
            self.dweights += self.weight_regularization_l1 * np.sign(self.weights)
        if self.bias_regularization_l1:
            self.dbias += self.bias_regularization_l1 * np.sign(self.biases)

        # L2 regularization
        if self.weight_regularization_l2:
            self.dweights += 2 * self.weight_regularization_l2 * self.weights
        if self.bias_regularization_l2:
            self.dbias += 2 * self.bias_regularization_l2 * self.biases
        self.dinputs = np.dot(self.weights.T, dvalues)


class LayerDropout(Base):
    def __init__(self, dropout_rate):
        self.success_rate = 1 - dropout_rate

    def forward(self, inputs):
        self.inputs = inputs
        self.binary_mask = (
            np.random.binomial(1, self.success_rate, inputs.shape) / self.success_rate
        )
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask
