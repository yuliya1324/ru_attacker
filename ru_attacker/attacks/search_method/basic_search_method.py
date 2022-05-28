from abc import ABC, abstractmethod

__all__ = ["BasicSearchMethod"]


class BasicSearchMethod(ABC):
    @abstractmethod
    def search(self, *args):
        pass
