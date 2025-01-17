# Small helper functions

from math import lgamma

import numpy as np

def _log_factorial(n):
    """Logarithm of factorial of n

    :param n: 
    :type n: float
    :return: log(n!)
    :rtype: float
    """
    
    return lgamma(n+1)

def _log_factorial2(n):
    """Logarithm of double factorial of n, for integer k, (2k)!! = k!2^k, (2k-1)!! = (2 k)!/(2^k k!)

    :param n: 
    :type n: float
    :return: log(n!!)
    :rtype: float
    """
    if n % 2 == 0:
        k = n/2
        return _log_factorial(k) + k * np.log(2)
    else:
        k = (n + 1)/2
        return _log_factorial(2*k) - _log_factorial(k) - k * np.log(2)

def _log_binom(n,m):
    """Logarithm of binomial coefficient binomial(n,m)

    :param n: 
    :type n: float
    :param m: 
    :type m: float
    :return: log(binomial(n,m))
    :rtype: float
    """
    
    return _log_factorial(n) - _log_factorial(m) - _log_factorial(n-m)

def _log_sum_exp(x):
    """Overflow protected log(sum(exp(x))) of an array x. 

    :param x: Array to be summed
    :type x: list
    :return: log(sum(exp(x)))
    :rtype: float
    """
    a = np.max(x)
    return a + np.log(np.sum(np.exp(x - a)))
