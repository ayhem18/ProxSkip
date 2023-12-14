from abc import ABC, abstractmethod

import numpy as np
from .model import Model
from .data import DataLoader
from .types import Vector


class LossFunction:
    def __init__(
        self,
        dataloader: DataLoader,
        model: Model
    ):
        self._dl = dataloader
        self._M = model

    
    @abstractmethod
    def dloss(self, j: int, size: int) -> Vector:
        pass
    
    @abstractmethod
    def loss(self, j: int, size: int) -> float:
        pass
    
    
    
class LogisticLoss(LossFunction):
    def __init__(self, dataloader: DataLoader, model: Model):
        super().__init__(dataloader, model)

    def dloss(self, j: int, size: int) -> Vector:
        X, y = self._dl.get(j, size)
        L = (np.linalg.norm(X, axis=1) ** 2 / 4).mean()
        lmbd = L / 1000
        
        w = self._M.params()
        g = X @ w
        s = y[:, None] * X
        s = s / (1 + np.exp(y * g))[:, None]
        return -s.mean(axis=0) + lmbd * w    
    
    
    