"""
This small script contains different optimizaiton functionalities: 
1. initial weight set up
2. algos
3. true solution
"""

import os
import pickle
import math
import random

import numpy as np

from tqdm import tqdm
from numpy.linalg import norm
from typing import Sequence, Union, List, Tuple
from scipy.optimize import minimize

from exp_setup import lr_loss


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

################################################## GENERAL UTILITIES #############################################

def find_x_true(all_data: np.ndarray, 
                all_labels: np.ndarray,
                lam: float) -> float:
    if not os.path.exists(os.path.join(SCRIPT_DIR, 'true_min.pkl')):
        # Finding true minimum to compare with
        def loss(w):
            return lr_loss(X=all_data, y=all_labels, w=w, lam=lam)

        res = minimize(loss, np.random.rand(all_data.shape[1]), method='BFGS', tol=10 ** -6)
        TRUE_MIN = res.fun
        pickle.dump(TRUE_MIN, open(os.path.join(SCRIPT_DIR, 'true_min.pkl'), 'wb'))

    else:
        with open(os.path.join(SCRIPT_DIR, 'true_min.pkl'), 'rb') as f:
            TRUE_MIN = pickle.load(f)

    return TRUE_MIN
    
def x_difference(x1: np.ndarray, x2: np.ndarray) -> float:
    """Computes the euclidean norm of the difference vector 

    Args:
        x1 (np.ndarray): first vector
        x2 (np.ndarray): second vector

    Returns:
        float: returns the norm of the difference
    """
    return norm(x1 - x2, ord=2)

def f_difference(f1: float, f2: float) -> float:
    """returns the absolute difference between 2 values """
    # the expression f_x_k - f_sol is equivalent since the problem is minimization,
    # but the 'abs' function was used to make the method general and not only specific to the given problem
    return abs(f1 - f2)      

# let's define the modes in terms of strings
X_DIFF = 'x_k+1 - x_k'
F_DIFF = 'f(x_k+1) - f(x_k)'
NORMALIZED_CRITERION = 'df_xk / df_x0'
X_OPT_DIFF = 'x* - x_k'


###################################################### LOCAL GD #################################################

def localGD(
    num_local_steps:int,
    device_data: Sequence[np.ndarray],
    device_labels: Sequence[np.ndarray],
    function: callable,
    gradient_function: callable,
    x_0: np.ndarray,
    x_sol: np.ndarray=None,
    K: int = 10 ** 3,
    eps: float = 10 ** -5, 
    mode: str = NORMALIZED_CRITERION,
    gamma_k: callable = None,                      
    return_history: bool = True
    ) -> Union[List[np.ndarray], np.ndarray]:

    # the first step is to make sure the 'mode' variable is defined above
    if isinstance(mode , str) and mode not in [F_DIFF, X_DIFF, NORMALIZED_CRITERION, X_OPT_DIFF]:
        raise ValueError((f"the variable 'mode' is expected to belong to {[F_DIFF, X_DIFF, NORMALIZED_CRITERION, F_DIFF]}"
                          f"Found: {mode}"))
    
    if mode == X_OPT_DIFF and x_sol is None: 
        raise ValueError(f"using mode = {X_OPT_DIFF} requires passing the solution to the problem")

    # make sure the number of local_steps is at least 1
    num_local_steps = max(1, num_local_steps)

    x_current = np.expand_dims(x_0) if x_0.ndim == 1 else x_0
    x_history = [x_current]
    criterion_history = []

    # first initialize all the devices with the same x0
    xts = [x_0.copy() for _ in range(len(device_data))]

    for k in tqdm(range(math.ceil(K / num_local_steps))):
        x_previous = x_current.copy()
        gamma = gamma_k(k)

        # iterate through each of the devices
        for device_index, (d_data, d_label) in enumerate(zip(device_data, device_labels)):
            # perform consecutive 'num_local_steps' updates
            for _ in range(num_local_steps):
                xts[device_index] = xts[device_index] - gamma * gradient_function(d_data, d_label, xts[device_index]) 

        # average the local weights
        avg_xt = np.mean(xts, axis=0).reshape(-1, 1)

        if avg_xt.shape != (device_data[0].shape[1], 1):
            raise ValueError(f"Make sure the average operation is carried successfully, mf !!. Found the following shape: {avg_xt.shape}")

        # set the local weights to the averaged weight
        xts = [avg_xt.copy() for _ in xts]

        # set the 'x_current' to the averged weight
        x_current = avg_xt.copy()

        if mode == F_DIFF:
            x_current_function_value = np.mean([function(d_data, d_label, x_current) for d_data, d_label in zip(device_data, device_labels)])
            x_previous_function_value = np.mean([function(d_data, d_label, x_previous) for d_data, d_label in zip(device_data, device_labels)])
            diff = f_difference(x_current_function_value, x_previous_function_value)
        
        elif mode == X_DIFF:
            diff = x_difference(x_current, x_previous)
        
        elif mode == NORMALIZED_CRITERION:
            x0_grad = np.mean([gradient_function(d_data, d_label, x_0) for d_data, d_label in zip(device_data, device_labels)], axis=0).reshape(-1, 1)
            current_grad = np.mean([gradient_function(d_data, d_label, x_current) for d_data, d_label in zip(device_data, device_labels)], axis=0).reshape(-1, 1)
            diff = norm(current_grad) / norm(x0_grad)
        
        elif mode == X_OPT_DIFF: 
            diff = norm(x_current - x_sol, ord=2)
        
        else: 
            # the last case is where the criterion is passed as an argument
            diff = mode(x_current)

        criterion_history.append(diff)
        # add 'x_current' to the history
        x_history.append(x_current)
            
        if diff <= eps: 
            break

        assert len(x_history) == k + 2, f"expected {k + 2} points. Found: {len(x_history)}"

    return  (x_history, criterion_history) if return_history else x_history[-1]


###################################################### ProxSKIP #################################################

def setup_hts_prox_skip(shape: Tuple, number: int):
    np.random.seed(55)
    hts = [np.random.randn(*shape) for _ in range(number)]
    avg = np.mean(hts, axis=0)    
    assert np.allclose(np.sum(np.asarray([h - avg for h in hts]), axis=0), 0), "The sum of hts do not add up to a 'zero' vector."
    return [h - avg for h in hts]



def proxSkipFL(
    devices_data,
    devices_labels,
    function: callable,
    gradient_function: callable,
    skip_probability: float,
    communication_rounds: int,
    max_iterations:int,
    x_0: np.ndarray,
    x_sol: np.ndarray=None,
    eps: float = 10 ** -5, 
    mode: str = NORMALIZED_CRITERION,
    gamma_k: callable = None,                      
    return_history: bool = True, 
    report_by_prox: int = 10):
    
    # set the seed so the prox executions are reproducible
    random.seed(69)
    # the first step is to make sure the 'mode' variable is defined above
    if isinstance(mode , str) and mode not in [F_DIFF, X_DIFF, NORMALIZED_CRITERION, X_OPT_DIFF]:
        raise ValueError((f"the variable 'mode' is expected to belong to {[F_DIFF, X_DIFF, NORMALIZED_CRITERION, X_OPT_DIFF]}"
                          f"Found: {mode}"))
    
    if mode == X_OPT_DIFF and x_sol is None: 
        raise ValueError(f"using mode = {X_OPT_DIFF} requires passing the solution to the problem")

    hts = setup_hts_prox_skip((devices_data[0].shape[1], 1), len(devices_data))
    xts = [x_0.copy() for _ in range(len(devices_data))]

    x_current = np.expand_dims(x_0) if x_0.ndim == 1 else x_0
    x_history = [x_current]
    criterion_history = []
    cs = 0

    for k in tqdm(range(max_iterations)):
        x_previous = x_current.copy()
        # find the value of gamma
        gamma = gamma_k(k)

        # iterate through the data
        for index, (d_data, d_labels) in enumerate(zip(devices_data, devices_labels)):
            xts[index] = xts[index] - gamma * (gradient_function(d_data, d_labels, xts[index]) - hts[index])

        # decide whether to prox or not
        on_prox = random.random() < skip_probability

        if on_prox:
            # calculate the average xt
            avg_xt = np.mean(xts, axis=0).reshape(-1, 1)
            # avg_xt = np.concatenate(xts, axis=1).mean(axis=1).reshape(-1, 1)

            assert avg_xt.shape == (devices_data[0].shape[1], 1), f"Make sure the averaging operation is correct. Found shape: {avg_xt.shape}"

            # update the controle variate
            for index in range(len(hts)):
                hts[index] = hts[index] + (skip_probability / gamma) * (avg_xt - xts[index])
            
            # set xts
            xts = [avg_xt.copy() for _ in xts]

            x_current = avg_xt.copy()
            # update the control variate
            
            if mode == F_DIFF:
                x_current_function_value = np.mean([function(d_data, d_labels, x_current) for d in devices_data])
                x_previous_function_value = np.mean([function(d_data, d_labels, x_previous) for d in devices_data])
                # the value of the function at 'x_current' is the average of the values of the local functions both at 'x_current' and 'x_previous'
                diff = f_difference(x_current_function_value, x_previous_function_value)
            
            elif mode == X_DIFF:
                diff = x_difference(x_current, x_previous)
            
            elif mode == NORMALIZED_CRITERION:
                # calculate the gradient at 'x_current' which is already calculated
                grad_x0 = np.concatenate(
                    [gradient_function(d_data, d_labels, x_0) for d_data, d_labels in zip(devices_data, devices_labels)]
                    ).mean(axis=1).reshape(-1, 1)
                
                grad_x_current = np.concatenate([gradient_function(d_data, d_labels, x_current) 
                                                        for d_data, d_labels in zip(devices_data, devices_labels)]).mean(axis=1).reshape(-1, 1)
                
                diff = norm(grad_x_current) / norm(grad_x0)
            
            elif mode == X_OPT_DIFF: 
                diff = norm(x_current - x_sol, ord=2)
            
            else: 
                # the last case is where the criterion is passed as an argument
                diff = mode(x_current)

            # if cs % report_by_prox == 0:
            #     print(f"Communication conducted: {cs + 1} times")
            
            # update the prox counter
            cs += 1

            criterion_history.append(diff)
            # add 'x_current' to the history
            x_history.append(x_current)
                
            if diff <= eps: 
                break
            
            if communication_rounds == cs :
                break

    return  (x_history, criterion_history) if return_history else x_history[-1]
