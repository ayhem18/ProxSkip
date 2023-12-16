import numpy as np
from .types import ProximityOperator, Vector


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
