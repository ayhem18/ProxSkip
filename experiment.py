from typing import Callable

import numpy as np
from proxskip.data import DataLoader
from proxskip.types import ProximityOperator, Vector
from proxskip.model import Model
from proxskip.loss import LossFunction
from proxskip.optimizer import ProxSkip


class ConsesusProx(ProximityOperator):
    def __call__(
        self,
        x: Vector,
        state: dict
    ):
        x_h_tp1 = state['x_h_tp1']
        avg_weight = np.zeros(shape=x_h_tp1[0].shape)

        for local_weight in x_h_tp1: 
            avg_weight += local_weight

        avg_weight /= len(x_h_tp1)
        return avg_weight


class Linear(Model):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._W = np.random.randn(in_features, out_features)
        # self._b = np.random.randn(out_features)

    def params(self) -> Vector:
        # return np.concatenate([self._W.ravel(), self._b.ravel()])
        return self._W

    def forward(self, x: Vector) -> Vector:
        # the assumption here is that 
        if x.ndim > 2: 
            raise ValueError(f"Please make sure the input is 2 dimensionsal. Found: {x.ndim} dimensions")        

        batch_size, features = x.shape
        if features != self.in_features: 
            raise ValueError(f"The input is expected to have: {self.in_features} features")

        output = x @ self._W

        assert output.shape == (batch_size, self.out_features), \
            f"Make sure the output is of the correct shape. Found: {batch_size, self.out_features}. "\
            "Found: {x.shape}"
        return output

    
    def backward(self, x: Vector, upstream: Vector) -> Vector:
        output = self.forward(x)
        # make sure the upstream grad is of the exact same shape
        assert upstream.shape == output.shape, f"Make sure the upstream gradient has the same as the output of the forward pass. Found: {upstream.shape}, Expected: {output.shape}"
        
        # calculate 'grad'
        grad = x.T @ upstream
        # make sure the grad is of the exact same shape as the weights  
        assert grad.shape == self._W.shape, f"Make sure the shape of the grad is the same as the weights. Found: {grad.shape}. Expected: {self._W.shape}"

        return grad


    def update(self, params: Vector) -> None:
        # self._W = params[:self._W.size].reshape(self._W.shape)
        # self._b = params[self._W.size:]
        self._W = params.copy()

class SimpleDataset(DataLoader):
    def __init__(self, X, y):
        self._X = X
        self._y = y

    def total_size(self) -> int:
        return self._X.shape[0]

    def get(self) -> tuple[Vector, Vector]:
        return self._X, self._y
    

class MSE(LossFunction):
    # def dloss(self, j: int, size: int) -> Vector:
    #     """Gradient of MSE loss function w.r.t. model parameters"""
    #     # extract the data and the labels
    #     X, y = self._dl.get(j, size)
    #     # forward pass: predict
    #     y_hat = self._M.forward(X)

    #     # MSE LOSS will be defined as the average of the squared error between y_hat and y
    #     # so the gradient with respect to each element in 'y_hat' is the difference between y_hat_i and y_i divided by the number of elements in 'y' 
    #     mse_loss_grad = (y_hat - y) / y.size
    #     # why mean ???
    #     return self._M.backward(X, mse_loss_grad)# .mean(axis=0) 
        
    # def loss(self, j: int, size: int) -> float:
    #     """MSE loss function"""
    #     X, y = self._dl.get(j, size)
    #     y_hat = self._M.forward(X)
    #     return ((y_hat - y) ** 2).mean()
    
    def upstream_gradient(self, y: Vector, y_hat: Vector) -> float:
        return (y_hat - y) / y.size
    
    def loss(self, m: Model, X: Vector, y: Vector) -> Vector:
        y_hat = m.forward(X)
        return ((y_hat - y) ** 2).mean()
        
    
    
    
    
X = np.random.randn(20, 10)
y = X @ np.random.randn(10, 1)
    
devices = [
    (Linear(10, 1), SimpleDataset(X[:10], y[:10])),
    (Linear(10, 1), SimpleDataset(X[10:], y[10:])),
]
    
optimizer = ProxSkip(
    models=[model for model, _ in devices],
    dataloaders=[dataloader for _, dataloader in devices],
    loss=MSE,
    prox=ConsesusProx(),
    num_iterations=1000,
    learning_rate=0.002,
    batch_size=100,
    p=1
)

while True:
    on_prox = optimizer.step()
    if on_prox is None:
        print(optimizer._step['loss'][-1])
    
    
print(optimizer._step['x_t'])
