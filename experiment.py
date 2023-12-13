from typing import Callable

import numpy as np
from proxskip.types import ProximityOperator, Vector, Function, Session
from proxskip.algorithm import Algorithm

class L1Norm(ProximityOperator):
    def __call__(
        self, 
        x: Vector, 
        gamma: Vector = None, 
        ksi: Callable[[Vector], Vector] = None
    ):
        u = np.full_like(x, -gamma + 2 * np.random.rand() * gamma)
        u[x > gamma] = x[x > gamma] - gamma
        u[x < -gamma] = x[x < -gamma] + gamma
        return u
        
        
class MSEOverDataset(Function):
    def __init__(self, batch_size, dim) -> None:
        super().__init__()
        self.X = np.random.rand(batch_size, dim)
        self.alpha_ = np.random.rand()
        self.beta_ = np.random.rand() / 1E4
        self.y = self.alpha_ * self.X + self.beta_

    def __call__(self, x: Vector, fx: Vector = None):
        return (2 * self.X * (x[None, :] * self.X - self.y)).mean(axis=0)
    
    
    
fn = MSEOverDataset(10, 20)
prox = L1Norm()
al = Algorithm(
    fn,
    prox
)

al.new_session(
    Session(
        num_iterations=100,
        step_size=0.0001,
        probability=1/100,
        x0=np.random.randn(20),
        h0=np.random.randn(20)
    )
)

while (x := al.step()):
    al.update(*x)
    print(((al._parameters * fn.X - fn.y) ** 2).mean())