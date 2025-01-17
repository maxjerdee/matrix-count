from __future__ import annotations

import numpy as np
from pytest import approx
from matrix_count._util import _log_binom, _log_sum_exp, _log_factorial, _log_factorial2

def test_log_factorial():
    n = 5
    result = 1
    for i in range(1, n + 1, 1):
        result *= i
    assert _log_factorial(n) == approx(np.log(result))

def test_log_factorial2():
    n = 5
    result = 1
    for i in range(1, n + 1, 2):
        result *= i
    assert _log_factorial2(n) == approx(np.log(result))

    n = 6
    result = 1
    for i in range(2, n + 1, 2):
        result *= i
    assert _log_factorial2(n) == approx(np.log(result))

def test_log_binom():
    n = 5
    m = 2
    assert _log_binom(n, m) == approx(np.log(10))

def test_log_sum_exp():
    test_array = [0, 1, 2]
    result = 0
    for a in test_array:
        result += np.exp(a)
    result = np.log(result)
    assert _log_sum_exp(test_array) == approx(result)