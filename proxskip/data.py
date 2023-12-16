from abc import ABC, abstractmethod
from .types import Vector

class DataLoader(ABC):
    @abstractmethod
    def get(self) -> tuple[Vector, Vector]:
        pass
    
    @abstractmethod
    def total_size(self) -> int:
        pass
    
    def get_data(self, left: int, size: int) -> tuple[Vector, Vector]:
        return self.get()

