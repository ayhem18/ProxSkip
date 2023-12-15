from abc import ABC, abstractmethod
from typing import List, Type
import numpy as np

from .loss import LossFunction

from .data import DataLoader

from .types import Vector, ProximityOperator
from .model import Model


class Optimizer:
    def __init__(
        self,
        models: Model | List[Model],
        dataloaders: DataLoader | List[DataLoader],
        loss: Type[LossFunction],
        prox: ProximityOperator = None,
        *,
        num_iterations: int = 10000,
        learning_rate: float = 0.1,
        batch_size: int = 1000,
    ) -> None:

        if isinstance(models, Model):
            models = [models]
        if isinstance(dataloaders, DataLoader):
            dataloaders = [dataloaders]

        self.models_ = models
        self.dl_ = dataloaders
        self.loss_ = [loss(dl, model)
                      for dl, model in zip(dataloaders, models)]
        self.prox_ = prox

        self._num_iterations = num_iterations
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._total_size = self.dl_[0].total_size()
        self._num_devices = len(models)

        assert all(dl.total_size() == self._total_size for dl in dataloaders), \
            'All dataloaders must have the same total size'

        self._step = {
            't': 0,
            'x_t': [model.params() for model in self.models_],
            'h_t': [np.random.rand(*model.params().shape) for model in self.models_],
            'losses': [],
            'gamma': self._learning_rate,
        }

    @abstractmethod
    def step(self) -> Vector:
        pass

    @abstractmethod
    def update(self, x_tp1: Vector) -> None:
        pass


class ProxSkip(Optimizer):
    def __init__(
        self,
        models: Model | List[Model],
        dataloaders: DataLoader | List[DataLoader],
        loss: Type[LossFunction],
        prox: ProximityOperator = None,
        *,
        num_iterations: int = 10000,
        learning_rate: float = 0.1,
        batch_size: int = 1000,
        p: float = 1 / 100,
    ) -> None:
        super().__init__(
            models,
            dataloaders,
            loss,
            prox,
            num_iterations=num_iterations,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )
        self.p_ = p

    def step(self) -> Vector:
        # extract the 
        t = self._step['t']

        if t >= self._num_iterations:
            return None

        x_t = self._step['x_t']
        h_t = self._step['h_t']

        # Iteration over devices
        batch_idx = (t % self._total_size) // self._batch_size
        n = batch_idx * self._batch_size

        # Flip a coin to decide whether to carry out with the prox opreation or not
        to_prox = np.random.rand() > self.p_
        
        # calculate the gradients of the loss function
        # with respect to the parameters of each model
        current_gradients = [
            self.loss_[i].dloss(n, self._batch_size)
            for i in range(len(self.models_))
        ]
        
        # Quantities for the update on prox skip
        phi_ = self._learning_rate / self.p_
        iphi_ = 1 / phi_
        
        # calculate x^{t + 1} for every model / device: Line 3 in the algorithm
        x_h_tp1 = [
            x_t[j] - self._learning_rate * (current_gradients[j] - h_t[j])
            for j in range(len(self.models_))
        ]

        if to_prox:
            # in the federated learning setting, 
            # the prox operation is averaging the values at the local devices

            # step 1: average all x_h_tp1: since we do not know the exact shape
            avg_weight = np.zeros(shape=x_h_tp1[0].shape)

            for local_weight in x_h_tp1: 
                avg_weight += local_weight

            avg_weight /= len(x_h_tp1)

            # set all the values in x_h_tp1 to 'avg_weight'
            x_tp1 = [avg_weight for _ in x_h_tp1]

            # x_tp1 = [
            #     self.prox_(x_h_tp1[j] - phi_ * h_t[j], self._step)
            #     for j in range(len(self.models_))
            # ]

        else:
            x_tp1 = x_h_tp1

        # for i, model in enumerate(self.models_):
        #     model.update(x_tp1[i])

        h_tp1 = [
            h_t[j] + iphi_ * (x_tp1[j] - x_h_tp1[j])
            for j in range(len(self.models_))
        ]
        

        self._step['t'] += 1
        self._step['x_t'] = x_tp1
        self._step['h_t'] = h_tp1
        self._step['losses'].append([
            self.loss_[i].loss(n, self._batch_size)
            for i in range(len(self.models_))
        ])

        return x_tp1

    def update(self) -> None:
        for j, model in enumerate(self.models_):
            model.update(self._step['x_t'][j])
 