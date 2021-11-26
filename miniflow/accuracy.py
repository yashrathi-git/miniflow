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
