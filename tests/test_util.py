from __future__ import annotations

import numpy as np
import pytest

from matrix_count._util import (
    _log_binom,
    _log_factorial,
    _log_factorial2,
    _log_sum_exp,
    _log_weight,
)


def test_log_factorial():
    n = 5
    result = 1
    for i in range(1, n + 1, 1):
        result *= i
    assert _log_factorial(n) == pytest.approx(np.log(result))


def test_log_factorial2():
    n = 5
    result = 1
    for i in range(1, n + 1, 2):
        result *= i
    assert _log_factorial2(n) == pytest.approx(np.log(result))

    n = 6
    result = 1
    for i in range(2, n + 1, 2):
        result *= i
    assert _log_factorial2(n) == pytest.approx(np.log(result))


def test_log_binom():
    n = 5
    m = 2
    assert _log_binom(n, m) == pytest.approx(np.log(10))


def test_log_sum_exp():
    test_array = [0.0, 1.0, 2.0]
    result = 0
    for a in test_array:
        result += np.exp(a)
    result = np.log(result)
    assert _log_sum_exp(test_array) == pytest.approx(result)


def test_log_weight():
    A = np.array([[2, 2, 3], [2, 3, 4], [3, 4, 4]])
    alpha = 5
    result = 17.0292069
    assert _log_weight(A, alpha) == pytest.approx(result)
