from abc import ABC, abstractmethod
from .types import Vector

class DataLoader(ABC):
    @abstractmethod
    def get(self) -> tuple[Vector, Vector]:
        pass
    
    @abstractmethod
    def total_size(self) -> int:
        pass
