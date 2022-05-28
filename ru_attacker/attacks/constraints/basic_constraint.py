from abc import ABC, abstractmethod

__all__ = ["BasicConstraint"]


class BasicConstraint(ABC):
    @abstractmethod
    def check(self, original, transformed):
        pass
