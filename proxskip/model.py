from abc import ABC, abstractmethod
from .types import Vector


class Model(ABC):
    @abstractmethod
    def params(self) -> Vector:
        pass

    @abstractmethod
    def forward(self, x: Vector) -> Vector:
        pass

    @abstractmethod
    def update(self, params: Vector) -> None:
        pass
    
    @abstractmethod
    def backward(self, x: Vector, upstream: Vector) -> Vector:
        pass
    
    
