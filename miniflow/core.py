from abc import ABC, abstractmethod


class Base(ABC):
    @abstractmethod
    def forward(self, inputs):
        pass
