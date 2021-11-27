from abc import ABC, abstractmethod

import numpy as np

from miniflow.activations import ActivationSoftmax


class BaseLoss(ABC):
    def __init__(self):
        self.trainable_layers = None

    def calculate(self, model_out, y, *, include_regularization=True):
        losses = self.forward(model_out, y)
        mean_loss = np.mean(losses)
        if not include_regularization:
            return mean_loss
        return mean_loss, self.regularisation_loss()

    @abstractmethod
    def forward(self, model_out, y_true):
        pass

    def regularisation_loss(self):
        regularisation_loss = 0
        for layer in self.trainable_layers:
            if layer.weight_regularization_l1:
                regularisation_loss += layer.weight_regularization_l1 * np.sum(
                    np.abs(layer.weights)
                )
            if layer.weight_regularization_l2:
                regularisation_loss += layer.weight_regularization_l2 * np.sum(
                    layer.weights ** 2
                )
            if layer.bias_regularization_l1:
                regularisation_loss += layer.bias_regularization_l1 * np.sum(
                    np.abs(layer.biases)
                )
            if layer.bias_regularization_l2:
                regularisation_loss += layer.bias_regularization_l2 * np.sum(
                    layer.biases ** 2
                )
        return regularisation_loss


class CategoricalLossEntropy(BaseLoss):
    def forward(self, model_out, y_true):
        m = model_out.shape[1]

        # Because this is passed to log which don't have 0 in its domain
        y_pred_clipped = np.clip(model_out, a_min=1e-7, a_max=1 + 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[y_true, range(m)]
        else:
            # Because elements of y_true would be either 0 or 1
            correct_confidences = np.sum(y_true * model_out, axis=0)
        floss = -np.log(correct_confidences).reshape((1, m))
        return floss

    def backwards(self, dvalues, y_true):
        m = dvalues.shape[1]
        n_output = dvalues.shape[0]
        if len(y_true.shape) == 1:
            # np.eye returns a diagonal matrix with n x n
            # y_true is in the form of classes which are correct probability
            y_true = np.eye(n_output)[y_true].T
        self.dinputs = -y_true / dvalues
        # normalize because it will help later in the optimization
        self.dinputs = self.dinputs / m


class BinaryCrossEntropy(BaseLoss):
    def forward(self, model_out, y_true):
        m = model_out.shape[1]
        model_out = np.clip(model_out, 1e-7, 1 - 1e-7)
        loss = -(y_true * np.log(model_out) + (1 - y_true) * np.log(1 - model_out))
        # averaging the loss per-neuron for given training example
        loss = np.mean(loss, axis=0, keepdims=True)
        assert loss.shape == (1, m)
        return loss

    def backward(self, dvalues, y_true):
        m = y_true.shape[1]
        outputs = dvalues.shape[0]

        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        # while forward prop we are averaging the loss per neuron for given
        # training example
        self.dinputs = (
            -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs
        )
        self.dinputs = self.dinputs / m


class LossMeanSquare(BaseLoss):
    def forward(self, model_out, y_true):
        m = model_out.shape[1]
        loss = (model_out - y_true) ** 2
        loss = np.mean(loss, axis=0, keepdims=True)
        assert loss.shape == (1, m)
        return loss

    def backward(self, dvalues, y_true):
        m = y_true.shape[1]
        outputs = dvalues.shape[0]
        self.dinputs = -2 * (y_true - dvalues) / outputs
        self.dinputs = self.dinputs / m


class LossMeanAbsolute(BaseLoss):
    def forward(self, model_out, y_true):
        m = model_out.shape[1]
        loss = np.abs(model_out - y_true)
        loss = np.mean(loss, axis=0, keepdims=True)
        assert loss.shape == (1, m)
        return loss

    def backward(self, dvalues, y_true):
        m = y_true.shape[1]
        outputs = dvalues.shape[0]
        self.dinputs = np.sign(y_true - dvalues) / outputs
        self.dinputs = self.dinputs / m


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
