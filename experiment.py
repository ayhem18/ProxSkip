from typing import Callable

import numpy as np
from proxskip.data import DataLoader
from proxskip.types import ProximityOperator, Vector
from proxskip.model import Model
from proxskip.loss import LossFunction
from proxskip.optimizer import ProxSkip


class L1Norm(ProximityOperator):
    def __call__(
        self,
        x: Vector,
        state: dict
    ):
        return x
        gamma = state.get('gamma')
        u = np.full_like(x, -gamma + 2 * np.random.rand() * gamma)
        u[x > gamma] = x[x > gamma] - gamma
        u[x < -gamma] = x[x < -gamma] + gamma
        return u


class Linear(Model):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self._W = np.random.randn(in_features, out_features)
        # self._b = np.random.randn(out_features)

    def params(self) -> Vector:
        # return np.concatenate([self._W.ravel(), self._b.ravel()])
        return self._W

    def forward(self, x: Vector) -> Vector:
        return x @ self._W
    
    def backward(self, x: Vector, upstream: Vector) -> Vector:
        return upstream @ self._W.T

    def update(self, params: Vector) -> None:
        # self._W = params[:self._W.size].reshape(self._W.shape)
        # self._b = params[self._W.size:]
        self._W = params.copy()

class SimpleDataset(DataLoader):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self._X = np.random.randn(100, in_features)
        self._y = self._X @ np.random.randn(in_features, out_features)

    def total_size(self) -> int:
        return self._X.shape[0]

    def get(self, j: int, size: int) -> tuple[Vector, Vector]:
        return self._X[j:j+size], self._y[j:j+size]
    

class MSE(LossFunction):
    def __init__(self, dataloader: DataLoader, model: Model) -> None:
        self._dl = dataloader
        self._M = model

    def dloss(self, j: int, size: int) -> Vector:
        """Gradient of MSE loss function w.r.t. model parameters"""
        X, y = self._dl.get(j, size)
        y_hat = self._M.forward(X)
        return self._M.backward(X, y_hat - y).mean(axis=0)
        
    def loss(self, j: int, size: int) -> float:
        """MSE loss function"""
        X, y = self._dl.get(j, size)
        y_hat = self._M.forward(X)
        return ((y_hat - y) ** 2).mean()
    
    
optimizer = ProxSkip(
    models=Linear(10, 10),
    dataloaders=SimpleDataset(10, 10),
    loss=MSE,
    prox=L1Norm(),
    num_iterations=1000,
    learning_rate=0.001,
    batch_size=100,
    p=0.5
)

while (x_t := optimizer.step()):
    optimizer.update()
    print([x for x in optimizer._step['losses'][-1]][0])
    
print(optimizer._step['x_t'])
