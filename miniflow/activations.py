import numpy as np

from miniflow.core import Base
from miniflow.loss import CategoricalLossEntropy


class ActivationSoftmax(Base):
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=0, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=0, keepdims=True)

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        # Loop over single training set
        for idx, (single_out, single_dval) in enumerate(zip(self.output.T, dvalues.T)):
            softmax_out = single_out.reshape(-1, 1)
            dvalues_out = single_dval.reshape(1, -1)  # row matrix
            jacobin_matrix = np.diagflat(softmax_out) - np.dot(
                softmax_out, softmax_out.T
            )
            self.dinputs[:, idx] = np.dot(dvalues_out, jacobin_matrix)


class ActivationReLU(Base):
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        # zero gradient where values were negative
        self.dinputs[self.inputs <= 0] = 0


class CommonSoftmaxCrossEntropyLoss:
    def __init__(self):
        self.activation = ActivationSoftmax()
        self.loss = CategoricalLossEntropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        m = dvalues.shape[1]
        if len(y_true.shape) == 2:
            # we just need to correct class indexes
            y_true = np.argmax(y_true, axis=0)
        self.dinputs = dvalues.copy()

        self.dinputs[y_true, range(m)] -= 1
        self.dinputs /= m
