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
        loss: LossFunction,
        prox: ProximityOperator = None,
        *,
        num_iterations: int = 10000,
        learning_rate: float = 0.1,
        # batch_size: int = 1000,
    ) -> None:

        if isinstance(models, Model):
            models = [models]
        if isinstance(dataloaders, DataLoader):
            dataloaders = [dataloaders]

        self.models_ = models
        self.dl_ = dataloaders
        self.loss_ = loss
        self.prox_ = prox

        self._num_iterations = num_iterations
        self._learning_rate = learning_rate
        # self._batch_size = batch_size
        self._total_size = self.dl_[0].total_size()
        self._num_devices = len(models)

        assert all(dl.total_size() == self._total_size for dl in dataloaders), \
            'All dataloaders must have the same total size'

        w = self.models_[0].params()
        for i in range(len(self.models_)):
            self.models_[i].weights = w

        h_t = [np.random.randn(*model.params().shape) for model in self.models_]

        h_t = np.array(h_t)
        h_t = (h_t - h_t.mean(axis=0))
        assert np.allclose(np.sum(h_t, axis=0), np.zeros_like(h_t[0])), \
            'Initial h_t does not sum to zero'
        self._step = {
            't': 0,
            'x_t': [model.params() for model in self.models_],
            'h_t': h_t,
            'x_h_tp1': [],
            'losses': [],
            'loss': [],
            'gamma': self._learning_rate,
        }

    @abstractmethod
    def step(self) -> Vector:
        pass

    @abstractmethod
    def update(self, x_tp1: Vector) -> None:
        pass
    
    
class LocalGD(Optimizer):
    def __init__(
        self,
        models: Model | List[Model],
        dataloaders: DataLoader | List[DataLoader],
        loss: LossFunction,
        prox: ProximityOperator = None,
        *,
        num_iterations: int = 10000,
        learning_rate: float = 0.1,
        # batch_size: int = 1000,
        communication_rate: int = 100
    ) -> None:
        super().__init__(
            models,
            dataloaders,
            loss,
            None,
            num_iterations=num_iterations,
            learning_rate=learning_rate,
            # batch_size=batch_size,
        )
        self.comminucation_rounds_ = communication_rate
        

    def step(self) -> Vector:
        t = self._step['t']

        if t >= self._num_iterations:
            return None

        x_t = self._step['x_t']

       
        upstream_gradients = []
        for i, dl in enumerate(self.dl_):
            X, y = dl.get()
            upstream_gradients.append(
                self.loss_.upstream_gradient(self.models_[i], X, y)
            )

        current_gradients = []
        for i, model in enumerate(self.models_):
            X, y = self.dl_[i].get()
            current_gradients.append(
                model.backward(X, upstream_gradients[i])
            )
        

        x_tp1 = [
            x_t[j] - self._learning_rate * current_gradients[j]
            for j in range(len(self.models_))
        ]
        
        to_sync = t % self.comminucation_rounds_ == 0
        
        if to_sync:
            x_tp1 = np.mean(x_tp1, axis=0)
            x_tp1 = [x_tp1.copy() for _ in range(len(self.models_))]

        self._step['t'] += 1
        self._step['x_t'] = x_tp1

        for i, model in enumerate(self.models_):
            model.update(x_tp1[i])

        if to_sync:
            uni_model = self.models_[0]
            loss = 0
            for i, dl in enumerate(self.dl_):
                X, y = dl.get()
                loss += self.loss_.loss(uni_model, X, y)
            loss /= len(self.dl_)
            self._step['loss'].append(loss)

        return to_sync


class ProxSkip(Optimizer):
    def __init__(
        self,
        models: Model | List[Model],
        dataloaders: DataLoader | List[DataLoader],
        loss: LossFunction,
        prox: ProximityOperator = None,
        *,
        num_iterations: int = 10000,
        learning_rate: float = 0.1,
        # batch_size: int = 1000,
        p: float = 1 / 100,
    ) -> None:
        super().__init__(
            models,
            dataloaders,
            loss,
            prox,
            num_iterations=num_iterations,
            learning_rate=learning_rate,
            # batch_size=batch_size,
        )
        self.p_ = p

    def step(self) -> Vector:
        # extract the 
        t = self._step['t']

        if t >= self._num_iterations:
            return None

        x_t = self._step['x_t']
        h_t = self._step['h_t']

        # Flip a coin to decide whether to carry out with the prox opreation or not
        to_prox = np.random.rand() < self.p_
        
        # calculate the gradients of the loss function
        # with respect to the parameters of each model
        upstream_gradients = []
        for i, dl in enumerate(self.dl_):
            X, y = dl.get()
            upstream_gradients.append(
                self.loss_.upstream_gradient(self.models_[i], X, y)
            )

        current_gradients = []
        for i, model in enumerate(self.models_):
            X, y = self.dl_[i].get()
            current_gradients.append(
                model.backward(X, upstream_gradients[i])
            )
        
        # Quantities for the update on prox skip
        phi_ = self._learning_rate / self.p_
        iphi_ = 1 / phi_
        
        # calculate x^{t + 1} for every model / device: Line 3 in the algorithm
        x_h_tp1 = [
            x_t[j] - self._learning_rate * (current_gradients[j] - h_t[j])
            for j in range(len(self.models_))
        ]
        
        self._step['x_h_tp1'] = x_h_tp1

        if to_prox:
            # in the federated learning setting, 
            # the prox operation is averaging the values at the local devices

            # step 1: average all x_h_tp1: since we do not know the exact shape
            # avg_weight = np.zeros(shape=x_h_tp1[0].shape)

            # for local_weight in x_h_tp1: 
            #     avg_weight += local_weight

            # avg_weight /= len(x_h_tp1)

            # # set all the values in x_h_tp1 to 'avg_weight'
            # x_tp1 = [avg_weight for _ in x_h_tp1]

            x_tp1 = [
                self.prox_(x_h_tp1[j] - phi_ * h_t[j], self._step)
                for j in range(len(self.models_))
            ]

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
     
        for i, model in enumerate(self.models_):
            model.update(x_tp1[i])
            
        if to_prox:
            uni_model = self.models_[0]
            loss = 0
            for i, dl in enumerate(self.dl_):
                X, y = dl.get()
                loss += self.loss_.loss(uni_model, X, y)
            loss /= len(self.dl_)
            self._step['loss'].append(loss)

        return to_prox

    # def update(self) -> None:
    #     for j, model in enumerate(self.models_):
    #         model.update(self._step['x_t'][j])
 