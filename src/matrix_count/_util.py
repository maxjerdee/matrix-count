# Small helper functions
from __future__ import annotations

from math import lgamma

import numpy as np
from numpy.typing import ArrayLike


def _log_factorial(n: float) -> float:
    """Logarithm of factorial of n

    :param n:
    :type n: float
    :return: log(n!)
    :rtype: float
    """
    return float(lgamma(n + 1))


def _log_factorial2(n: float) -> float:
    """Logarithm of double factorial of n, for integer k, (2k)!! = k!2^k, (2k-1)!! = (2 k)!/(2^k k!)

    :param n:
    :type n: float
    :return: log(n!!)
    :rtype: float
    """
    if n % 2 == 0:
        k = n / 2
        return _log_factorial(k) + k * float(np.log(2))
    k = (n + 1) / 2
    return _log_factorial(2 * k) - _log_factorial(k) - k * float(np.log(2))


def _log_binom(n: float, m: float) -> float:
    """Logarithm of binomial coefficient binomial(n,m)

    :param n:
    :type n: float
    :param m:
    :type m: float
    :return: log(binomial(n,m))
    :rtype: float
    """
    return _log_factorial(n) - _log_factorial(m) - _log_factorial(n - m)


def _log_sum_exp(
    x: ArrayLike,
) -> float | np.complex64:
    """Overflow protected log(sum(exp(x))) of an array x.

    :param x: Array to be summed
    :type x: np.ndarray[np.float_, np.dtype[np.float_]]
    :return: log(sum(exp(x)))
    :rtype:
    """
    x = np.array(x)
    a: np.float64 | np.complex64 = np.max(x)
    return a + np.log(np.sum(np.exp(x - a)))


def _log_c(x: float) -> np.complex64:
    """Version of logarithm that complexifies arguments to deal with negative numbers

    :param x: float
    :type x: float
    :return: log(x)
    :rtype:
    """
    return np.log(x + 0j)


def _log_weight(A: ArrayLike, alpha: float) -> float:
    """Logarithm of the weight of a matrix A under the Dirichlet-multinomial distribution with parameter alpha

    :param A: Matrix
    :type A: ArrayLike
    :param alpha: Dirichlet-multinomial parameter
    :type alpha: float
    :return: log(weight)
    :rtype: float
    """
    log_weight = 0.0
    for i in range(A.shape[0]):
        for j in range(i + 1, A.shape[1]):
            log_weight += _log_binom(A[i, j] + alpha - 1, alpha - 1)
        log_weight += _log_binom(A[i, i] / 2 + alpha - 1, alpha - 1)

    return log_weight
