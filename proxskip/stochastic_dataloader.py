import math
import random
import numpy as np

from proxskip.data import DataLoader
from proxskip.types import Vector

random.seed(69)
np.random.seed(69)

class StochasticBatchedDataset(DataLoader):
    def __init__(self, 
                 data, 
                 labels: np.ndarray,
                 batch_size: int,
                 even_split: bool = False):

        num_samples = data.shape[0]
        if even_split and num_samples % batch_size != 0: 
            raise ValueError(f"if 'even_split' is set to True, the batches must of the same size")

        self.batch_size = batch_size
        self.num_batches = int(math.ceil(num_samples / batch_size))
        self.data = data.toarray()
        self.labels = labels

        num_samples, _ = self.data.shape
        # first calculate L (without lambda)
        self.L = np.linalg.norm(self.data) ** 2 / (4 * num_samples)


    def get(self, batch_index: int = None) -> tuple[Vector, Vector]:
        batch_index = (random.randint(0, self.num_batches - 1)) if batch_index is None else batch_index
        x, y = (self.data[batch_index * self.batch_size: (batch_index + 1) * self.batch_size], 
                self.labels[batch_index * self.batch_size: (batch_index + 1) * self.batch_size])
        return x, y

    def total_size(self) -> int:
        return self.data.shape[0]    
    
    def get_data(self, left: int = None, right: int = None) -> tuple[Vector, Vector]:
        # make sure both 'left' and 'right' are None or none of them is
        if (left is None) != (right is None):
            raise TypeError(f"either both 'left' and 'right' must be None. or None of them")

        if left is None:
            return self.data, self.labels
        
        return self.data[left: left + right], self.labels[left: left + right]