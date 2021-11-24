import numpy as np
from abc import ABC, abstractmethod


class BaseLoss(ABC):
    def calculate(self, model_out, y):
        losses = self.forward(model_out, y)
        mean_loss = np.mean(losses, axis=1)
        return np.squeeze(mean_loss)

    @abstractmethod
    def forward(self, model_out, y_true):
        pass


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
