from abc import ABC, abstractmethod

__all__ = ["BasicGoalFunction"]


class BasicGoalFunction(ABC):
    @abstractmethod
    def success(self, label, prediction):
        pass
