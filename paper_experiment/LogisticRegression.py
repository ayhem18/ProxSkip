"""
This script contains all the functionalities needed to calculate, optimize and work with the Logistic Regression problem.
"""


import numpy as np
import numdifftools as nd

from typing import Tuple, List, Union


def verify_input(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # first let's make sure the input is as expected: X is expected to have samples as rows
    if X.ndim != 2:
        raise ValueError(f"the training is expected to be 2 dimensional. Found: {X.shape}")
    
    num_samples, dim = X.shape 
    # make sure 'y' matches the number of samples
    y = np.expand_dims(y, axis=1) if y.ndim == 1 else y

    if y.shape != (num_samples, 1):
        raise ValueError((f"The number of labels is expected to match the number of samples"
                          f"\nFound: {y.shape} Expected: {(num_samples, 1)}"))

    # check 'w' as well:
    w = np.expand_dims(w, axis=-1) if w.ndim == 1 else w

    # make sure the dimensions match
    if w.shape != (dim, 1):
        raise ValueError((f"The weight matrix 'w' is expected as a column vector with length {dim}\n"
                          f"Expected: {(dim, 1)}. Found: {w.shape}"))
    return X, y, w


def logistic_regression_loss(
                    X: np.ndarray, 
                    y: np.ndarray,
                    w: np.ndarray, 
                    lam: float) -> float:
    """ This function calculates different values of the function gives it parameters:

    Args:
        X (np.ndarray): The training data
        y (np.ndarray): The labels
        w (np.ndarray): the weights
        lam (float): lambda: the regularization hyper-parameter

    Returns:
        float: The value for the function with parameter 'w'
    """
    # first step is to verify the input
    X, y, w = verify_input(X, y, w)
        
    A = X * y
    return np.mean(np.log(1 + np.exp((- A @ w).astype(np.float64)))) + (lam / 2) * np.linalg.norm(w) ** 2 


def logistic_regression_gradient(X: np.ndarray, 
                      y: np.ndarray, 
                      w: np.ndarray, 
                      lam: float) -> np.ndarray:
    # verify the input
    X, y, w = verify_input(X, y, w)

    A = X * y
    preds = A @ w 

    # preds contains the w^T x_i in each row
    # apply the exponential on each
    # A / (1 + np.exp(preds)) contains y_i * x_i / (1 + e^{w_i^T y_i x_i}) in each row: which a row vector

    # then calculate the mean over axis=0 to get a row vector
    dl1 = np.mean(A / (1 + np.exp(preds)), axis=0)

    assert dl1.shape == (X.shape[1], )

    dl = - np.expand_dims(dl1, axis=-1) + lam * w

    assert dl.shape == (X.shape[1], 1), "The gradient is expected to be a column vector."
    
    return dl


def logistic_regression_hessian(X: np.ndarray, 
                   y: np.ndarray, 
                   w: np.ndarray, 
                   lam: float) -> np.ndarray:
    # verify the input
    X, y, w = verify_input(X, y, w)
    num_samples, dim = X.shape

    A = X * y
    preds = (A @ w).astype(np.float64) 
    # preds = preds.astype(np.float64)

    # create the hessian matrix with zero values
    hessian_matrix = np.zeros((dim, dim), dtype=np.float64)

    exp_coeffs = (1 / (1 + np.exp(preds))) - (1 / ((1 + np.exp(preds)) ** 2))

    assert exp_coeffs.shape == (num_samples, 1)

    #iterate through samples
    for i in range(num_samples):
        # first step calculate the x_i * x_i ^ T
        # extract x_i: row vector in code
        x_i = X[i, :]
        
        assert x_i.shape == (dim,) , "The row sample is expected to be 1 dimensional"
        # expand 
        x_i = np.expand_dims(x_i, axis=-1)

        matrix_xi = x_i @ x_i.T
        # add an assert to catch any errors with shapes
        assert matrix_xi.shape == (dim, dim), "Make sure the matrix x_i * x_i ^ T is computed correctly"

        hessian_xi = exp_coeffs[i][0] * matrix_xi
        # time to add the coefficient associated with the matrix_xi
        # hessian_xi = (exp1[i][0] / exp2[i][0]) * matrix_xi
        hessian_matrix += hessian_xi

    # make sure to divide by the number of samples 
    hessian_matrix = hessian_matrix / num_samples + lam * np.eye(dim)
    # make sure the shape of the hessian matrix 
    hessian_matrix.shape == (dim , dim), "Make sure the hessian matrix is of the correct shape"
    return hessian_matrix


def logistic_regression_L_estimation(X: np.ndarray,
                    y: np.ndarray,
                    w: np.ndarray) -> float:
    # return the estimation by the problem description
    X, y, w = verify_input(X, y, w) 
    num_samples, _ = X.shape
    # first calculate L (without lambda)
    L = np.linalg.norm(X) ** 2 / (4 * num_samples)
    # set lambda
    return L

# add some tests:
# let's write a few tests for our function: 
import random

def test_logistic_regression():
    for _ in range(100):
        n, dim = random.randint(10, 20), random.randint(3, 10)    
        # test the case where all x = 1 and y = 1 
        x = np.ones(shape=(n, dim))        
        y = np.ones(shape=(n, 1))        
        lam = max(random.random(), 0) + 10 ** -4
        w = np.random.rand(dim, 1)
        
        result = np.log(1 + np.exp(-np.sum(w))) + (lam / 2) * np.linalg.norm(w) ** 2

        v = logistic_regression_loss(x, y, w , lam)

        assert np.isclose(result, v), f"the value function failed for {w} with case x = 1, y = 1"

        # test the case where all x = 1 and y = -1
        x = np.ones(shape=(n, dim))        
        y = -np.ones(shape=(n, 1))        
        lam = max(random.random(), 0) + 10 ** -4
        w = np.random.rand(dim, 1)
        result = np.log(1 + np.exp(np.sum(w))) + (lam / 2) * np.linalg.norm(w) ** 2
        v = logistic_regression_loss(x, y, w , lam)
        
        assert np.isclose(result, v), f"the value function failed for {w} with case x = 1, y = -1"
        
        # test the case where x = 0
        x = np.zeros(shape=(n, dim))
        y = np.ones(shape=(n, 1))
        lam = max(random.random(), 0) + 10 ** -4
        w = np.random.rand(dim, 1)
        result = np.log(2) + (lam / 2) * np.linalg.norm(w) ** 2
        v = logistic_regression_loss(x, y, w , lam)

        assert np.isclose(result, v), f"The value function failed for {w} with case x = 0"

        # the hardest case:     
        dim =  random.randint(10, 25)
        # test the code with x_i = e_i (zeroes at all positions) but i-th entry (with a one)
        x = np.eye(dim)
        y = np.ones(shape=(dim, 1))
        lam = max(random.random(), 0) + 10 ** -4
        w = np.random.rand(dim, 1)

        result = np.mean([np.log(1 + np.exp(-w[i][0])) for i in range(dim)]) + (lam / 2) * np.linalg.norm(w) ** 2
        v = logistic_regression_loss(x, y, w , lam)

        assert np.isclose(result, v), f"The value function failed with case x_i = e_i"

# test the gradient

def test_gradient_function():
    for _ in range(100):
        # generate random 'w'
        n, dim = random.randint(10, 20), random.randint(3, 10)    
        # test the case where all x = 1 and y = 1 
        x = np.ones(shape=(n, dim))        
        y = np.ones(shape=(n, 1))        
        lam = max(random.random(), 0) + 10 ** -4

        # function
        f = lambda v: logistic_regression_loss(X=x, y=y, w=v, lam=lam)
        # gradient
        g = lambda v: logistic_regression_gradient(X=x, y=y, w=v, lam=lam)

        for i in range(100):
            w = np.random.rand(dim, 1)
            # calculate the gradient
            x1 = nd.Gradient(f)(w)
            x2 = np.squeeze(g(w))

            assert x1.shape == x2.shape, f"x1: {x1.shape}, x2: {x2.shape}"
            assert np.allclose(x1, x2, atol=10**-8), "The gradients are too different"


def test_hessian_function():
    for _ in range(100):
        # generate random 'w'
        n, dim = random.randint(10, 20), random.randint(3, 10)    
        # test the case where all x = 1 and y = 1 
        x = np.ones(shape=(n, dim))        
        y = np.ones(shape=(n, 1))        
        lam = max(random.random(), 0) + 10 ** -4

        # function
        f = lambda v: logistic_regression_loss(X=x, y=y, w=v, lam=lam)
        # hessian
        h = lambda v: logistic_regression_hessian(X=x, y=y, w=v, lam=lam)

        for i in range(100):
            w = np.random.rand(dim)
            # calculate the gradient
            x1 = nd.Hessian(f)(w)
            x2 = np.squeeze(h(w))

            assert x1.shape == x2.shape, f"x1: {x1.shape}, x2: {x2.shape}"
            assert np.allclose(x1, x2, atol=10**-6), "The Hessians are too different" 


if __name__ == '__main__':
    # test the loss
    print("Loss tests: started")
    test_logistic_regression()
    print("Loss tests: completed")

    print("gradient tests: started")
    test_gradient_function()
    print("gradient tests: completed")

    print("Hessian tests: started")
    test_hessian_function()
    print("Hessian tests: completed")
