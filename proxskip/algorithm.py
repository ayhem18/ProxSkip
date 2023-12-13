import numpy as np

from .types import Vector, Session, Function, ProximityOperator

class Algorithm:
    def __init__(
        self,
        nabla_f: Function,
        prox: ProximityOperator
    ) -> None:
        self._parameters = None
        self._shape = None
        self._session = None
        
        self._nf = nabla_f
        self._prox = prox
        self._h = None
    
    def _sample_h0(self):
        return np.random.rand(self._shape)
    
    
    def new_session(
        self,
        session: Session
    ) -> None: 
        if session.h0 is None:
            session.h0 = self._sample_h0()
        
        if session.x0 is not None:
            self._parameters = session.x0.copy()
            self._shape = self._parameters.shape
        else:
            session.x0 = self._parameters.copy()
            
        self._session = session
        self._h = session.h0
    
        
    
    def step(
        self,
    ):
        if self._session.current_step >= self._session.num_iterations:
            return None
        self._session.current_step += 1
        x_t = self._parameters.copy()
        h_t = self._h.copy()
        p = self._session.probability
        gamma = self._session.step_size
        
        # \hat x_{t + 1}
        x_h_tp1 = x_t - gamma * (self._nf(x_t) - h_t)
        
        x_tp1 = None # x_{t + 1}
        h_tp1 = None # h_{t + 1}
        
        # Flip a coin \theta \in \{0, 1 \} where Prob(\theta_{t} = 1) = p
        theta = (np.random.rand() < p).all()
        
        if theta:
            x_tp1 = self._prox(x_h_tp1 - gamma / p * h_t, gamma=gamma)
        else: # Skip prox!
            x_tp1 = x_h_tp1
            
        h_tp1 = h_t + p / gamma * (x_tp1 - x_h_tp1)
        
        return x_tp1, h_tp1
    
    def update(
        self,
        x_tp1: Vector,
        h_tp1: Vector,
    ):
        self._h = h_tp1
        self._parameters = x_tp1
        