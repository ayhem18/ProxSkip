from abc import ABC, abstractmethod
from .types import Vector

class DataLoader(ABC):
    @abstractmethod
    def get(index: int, size: int) -> tuple[Vector, Vector]:
        pass
    
    @abstractmethod
    def total_size() -> int:
        pass
