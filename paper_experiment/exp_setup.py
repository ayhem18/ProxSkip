"""
This is a small script with to download, split and compute different needed to replicate the experiments in the paper.
"""

import os
import numpy as np
import urllib
import random

from typing import Union, Tuple, Sequence
from sklearn import datasets
from optmethods.loss import LogisticRegression    

# set the random see
np.random.seed(69)
random.seed(69)

def download_dataset() -> Tuple:
    DATASET_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w8a"
    data_path = './w8a'
    if not os.path.exists(data_path):
        f = urllib.request.urlretrieve(DATASET_URL, data_path)
    d, l = datasets.load_svmlight_file(data_path)
    l = l * (l == 1)
    return d, l 


def split_into_batches(X: np.ndarray,
                       batch_size: int, 
                       y: np.ndarray=None,
                       even_split: bool = True
                       ) -> Union[Tuple[Sequence[np.ndarray], Sequence[np.ndarray]], Sequence[np.ndarray]]:
    """A function to split the data into batches. I discard shuffling in this function to make sure 
    the output is deterministic for the same 'X'

    Args:
        X (np.ndarray): The data samples
        batch_size (int): 
        y (np.ndarray, optional): labels. Defaults to None.
        even_split (bool, optional): Whether to have all batches split evenly. Defaults to True.

    Returns:
        Union[Tuple[Sequence[np.ndarray], Sequence[np.ndarray]], Sequence[np.ndarray]]: the data batches, [Optional] the label batches 
    """
    # set the seed for reproducibility
    np.random.seed(31)

    # shuffle the data
    index = np.arange(X.shape[0])
    np.random.shuffle(index)
    X = X[index]
    y = y[index]
    
    # convert the batch size to 'int'
    batch_size = int(batch_size)
    
    # make sure to raise an error if the number of samples cannot be split into even batches (in case of 'even_split' is True)
    if even_split and X.shape[0] % batch_size != 0:
        raise ValueError((f"Please pass a batch size that can split the data evenly or set 'even_split' to False.\n" 
                         f"The number of samples: {len(X)}. The batch size: {batch_size}"))
    
    if y is not None: 
        # make sure the number of samples is the same as that of the number of labels
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"The number of samples should be the same as the number of labels.")

        # y = np.expand_dims(y, axis=-1) if y.ndim == 1 else y        

    if y is not None:
        return [X[i: i + batch_size] for i in range(0, X.shape[0], batch_size)], [y[i: i + batch_size] for i in range(0, y.shape[0], batch_size)]

    return [X[i: i + batch_size] for i in range(0, X.shape[0], batch_size)]



def L_estimation(device_batches: Sequence[np.ndarray], 
                 label_batches: Sequence[np.ndarray]) -> float:
    """
    This function estimates 'L' such that each inner function f_i is 'L' smooth.
    Args:
        device_batches (Sequence[np.ndarray]): The data splits assigned to each device
        label_batches (Sequence[np.ndarray]): The label splits assigned to each device
    Returns:
        float: The estimation of the smoothness constant
    """

    our_l = float('-inf')
    for d_data, d_label in zip(device_batches, label_batches):
        loss = LogisticRegression(d_data, d_label, l1=0, l2=0)
        l = loss.smoothness
        our_l = max(our_l, l)
    return our_l            


def lr_loss(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam) -> float:
    return LogisticRegression(X, y, l1=0, l2=lam).value(w)


def lr_gradient(X: np.ndarray, y:np.ndarray, w: np.ndarray, lam: float) -> np.ndarray:
    return LogisticRegression(X, y, l1=0, l2=lam).gradient(w.squeeze()).reshape(-1, 1)  

        
def stochastic_lr_loss(X: np.ndarray, y: np.ndarray, w: np.ndarray, batch_size: int, lam: float) -> float:
    batch_index = random.randint(0, X.shape[0] - batch_size - 1)
    return LogisticRegression(X[batch_index: batch_index + batch_size], y[batch_index: batch_index + batch_size], l1=0, l2=lam).value(w)


def stochastic_lr_gradient(X: np.ndarray, y:np.ndarray, w: np.ndarray, batch_size: int, lam: float) -> np.ndarray:
    batch_index = random.randint(0, X.shape[0] - batch_size - 1)    
    return LogisticRegression(X[batch_index: batch_index + batch_size], y[batch_index: batch_index + batch_size], l1=0, l2=lam).gradient(w.squeeze()).reshape(-1, 1) 
