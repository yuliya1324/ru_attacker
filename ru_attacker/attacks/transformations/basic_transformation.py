from abc import ABC, abstractmethod

__all__ = ["BasicTransformation"]


class BasicTransformation(ABC):
    @abstractmethod
    def transform(self, text):
        pass
