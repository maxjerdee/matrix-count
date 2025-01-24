# Small helper functions
from __future__ import annotations

from math import lgamma

import numpy as np
from numpy.typing import ArrayLike


def log_factorial(n: float) -> float:
    """
    Logarithm of factorial of n.

    Parameters
    ----------
    n : float
        Input value.

    Returns
    -------
    float
        log(n!).
    """
    return float(lgamma(n + 1))


def log_factorial2(n: float) -> float:
    """
    Logarithm of double factorial of n, for integer k, (2k)!! = k!2^k, (2k-1)!! = (2 k)!/(2^k k!).

    Parameters
    ----------
    n : float
        Input value.

    Returns
    -------
    float
        log(n!!).
    """
    if n % 2 == 0:
        k = n / 2
        return log_factorial(k) + k * float(np.log(2))
    k = (n + 1) / 2
    return log_factorial(2 * k) - log_factorial(k) - k * float(np.log(2))


def log_binom(n: float, m: float) -> float:
    """
    Logarithm of binomial coefficient binomial(n,m).

    Parameters
    ----------
    n : float
        Total number of items.
    m : float
        Number of items to choose.

    Returns
    -------
    float
        log(binomial(n,m)).
    """
    return log_factorial(n) - log_factorial(m) - log_factorial(n - m)


def log_sum_exp(x: ArrayLike) -> float | np.complex64:
    """
    Overflow protected log(sum(exp(x))) of an array x.

    Parameters
    ----------
    x : ArrayLike
        Array to be summed.

    Returns
    -------
    float or np.complex64
        log(sum(exp(x))).
    """
    x = np.array(x)
    a: np.float64 | np.complex64 = np.max(x)
    return a + np.log(np.sum(np.exp(x - a)))


def log_c(x: float) -> np.complex64:
    """
    Version of logarithm that complexifies arguments to deal with negative numbers.

    Parameters
    ----------
    x : float
        Input value.

    Returns
    -------
    np.complex64
        log(x).
    """
    return np.log(x + 0j)


def log_weight(A: ArrayLike, alpha: float) -> float:
    """
    Logarithm of the weight of a matrix A under the Dirichlet-multinomial distribution with parameter alpha.

    Parameters
    ----------
    A : ArrayLike
        Matrix.
    alpha : float
        Dirichlet-multinomial parameter.

    Returns
    -------
    float
        log(weight).
    """
    result = 0.0
    for i in range(A.shape[0]):
        for j in range(i + 1, A.shape[1]):
            result += log_binom(A[i, j] + alpha - 1, alpha - 1)
        result += log_binom(A[i, i] / 2 + alpha - 1, alpha - 1)

    return result
