from abc import ABC, abstractmethod

import numpy as np


class BaseAccuracy(ABC):
    def calculate(self, y_pred, y_true):
        comparisons = self.compare(y_pred, y_true)
        accuracy = np.mean(comparisons)
        return accuracy

    @abstractmethod
    def compare(self, y_pred, y_true):
        raise NotImplementedError("To be implemented by subclass")

    @abstractmethod
    def init(self, y, reinit=False):
        raise NotImplementedError("To be implemented by subclass")


class AccuracyRegression(BaseAccuracy):
    def __init__(self, precision=None):
        self.precision = precision

    def init(self, y, reinit=False):
        if (self.precision is None) or reinit:
            self.precision = np.std(y) / 250

    def compare(self, y_pred, y_true):
        return np.absolute(y_pred - y_true) < self.precision


class AccuracyCategorical(BaseAccuracy):
    def compare(self, y_pred, y_true):
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=0)
        assert len(y_pred.shape) == 1
        assert len(y_true.shape) == 1
        return y_pred == y_true

    def init(self, y, reinit=False):
        pass
