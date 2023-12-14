from abc import ABC, abstractmethod
from typing import Optional, List, Callable
import numpy as np

Vector = np.ndarray


class Session:
    num_iterations: int
    step_size: np.float32
    probability: np.float32
    x0: Optional[np.float32] = None
    h0: Optional[np.float32] = None
    errs: List[Vector] = []
    current_step: int
    
    def __init__(
        self,
        num_iterations: int,
        step_size: np.float32,
        probability: np.float32,
        x0: np.float32,
        h0: np.float32,
    ) -> None:
        self.num_iterations = num_iterations
        self.step_size = step_size
        self.probability = np.float32(probability)
        self.x0 = x0
        self.h0 = h0
        self.current_step = 0


class ProximityOperator(ABC):
    @abstractmethod
    def __call__(
        self,
        x: Vector,
        gamma: Vector = None,
        ksi: Callable[[Vector], Vector] = None
    ):
        pass


class Function(ABC):
    @abstractmethod
    def __call__(
        self,
        x: Vector,
        fx: Vector = None
    ):
        pass
