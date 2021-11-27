import numpy as np

from miniflow.common import Base


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

    @staticmethod
    def predictions(outputs):
        # Taking the index of maximum activated neuron - its class
        return np.argmax(outputs, axis=0)


class ActivationReLU(Base):
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        # zero gradient where values were negative
        self.dinputs[self.inputs <= 0] = 0

    @staticmethod
    def predictions(outputs):
        return outputs


class ActivationSigmoid(Base):
    def forward(self, inputs: np.ndarray):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output

    @staticmethod
    def predictions(outputs):
        return np.round(outputs)


class ActivationLinear(Base):
    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

    @staticmethod
    def predictions(outputs):
        return outputs
