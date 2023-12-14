from abc import ABC, abstractmethod
from typing import Optional, List, Callable
import numpy as np

Vector = np.ndarray

class ProximityOperator(ABC):
    @abstractmethod
    def __call__(
         self,
         x: Vector,
         state: dict
    ) -> Vector:
         pass


class Model(ABC):
    @abstractmethod
    def call_f(
        self,
        x: Vector,
    ):
        pass
    
    @abstractmethod
    def call_df(
        self,
        x: Vector,
    ) -> None:
        pass