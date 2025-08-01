from __future__ import annotations

import numpy as np
import pytest

from matrix_count._util import (
    _erdos_gallai_check,
    _log_binom,
    _log_factorial,
    _log_factorial2,
    _log_sum_exp,
    _log_weight,
    _lgamma_c,
    _log_P_dirichlet_multinomial,
)


def test__log_factorial():
    n = 5
    result = 1
    for i in range(1, n + 1, 1):
        result *= i
    assert _log_factorial(n) == pytest.approx(np.log(result))


def test__log_factorial2():
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


def test__log_binom():
    n = 5
    m = 2
    assert _log_binom(n, m) == pytest.approx(np.log(10))


def test__log_sum_exp():
    test_array = [0.0, 1.0, 2.0]
    result = 0
    for a in test_array:
        result += np.exp(a)
    result = np.log(result)
    assert _log_sum_exp(test_array) == pytest.approx(result)


def test__log_weight():
    A = np.array([[2, 2, 3], [2, 3, 4], [3, 4, 4]])
    alpha = 5
    result = 17.0292069
    assert _log_weight(A, alpha) == pytest.approx(result)


def test__erdos_gallai_check():
    ks = np.array([4, 2, 2, 1, 1])
    assert _erdos_gallai_check(ks)

    ks = np.array([4, 3, 1, 1, 1])
    assert not _erdos_gallai_check(ks)

    ks = np.array([13, 5, 14, 5, 5, 15, 9, 3, 1, 6, 2, 7, 2, 6, 6, 1])
    assert not _erdos_gallai_check(ks)


def test__lgamma_c():
    n = 1.5
    result = -0.120782 + 0j
    test_result = _lgamma_c(n)
    assert np.abs(test_result) == pytest.approx(np.abs(result), abs=1e-5)
    assert np.angle(test_result) == pytest.approx(np.angle(result), abs=1e-5)

    n = -1.5
    result = 0.860047 + 0j
    test_result = _lgamma_c(n)
    assert np.abs(test_result) == pytest.approx(np.abs(result), abs=1e-5)
    assert np.angle(test_result) == pytest.approx(np.angle(result), abs=1e-5)

    n = -2.5
    result = -0.0562437 - 3.14159j
    test_result = _lgamma_c(n)
    assert np.abs(test_result) == pytest.approx(np.abs(result), abs=1e-5)
    assert np.angle(test_result) == pytest.approx(np.angle(result), abs=1e-5)


def test__log_P_dirichlet_multinomial():
    ks = np.array([1,2,3])
    alpha = 1.5
    result = -3.0908
    assert _log_P_dirichlet_multinomial(ks, alpha) == pytest.approx(result, 0.001)

    ks = np.array([1,2,3])
    alpha = -1.5
    result = 0.538997
    assert _log_P_dirichlet_multinomial(ks, alpha) == pytest.approx(result, 0.001)

    ks = np.array([1,2,3])
    alpha = np.inf
    result = -2.49733
    assert _log_P_dirichlet_multinomial(ks, alpha) == pytest.approx(result, 0.001)

    ks = np.array([0,0,6])
    alpha = -1.5
    result = -1.09861
    assert _log_P_dirichlet_multinomial(ks, alpha) == pytest.approx(result, 0.001)

    ks = np.array([1,1,1,1,2])
    alpha = -2.6
    result = -2.89334
    assert _log_P_dirichlet_multinomial(ks, alpha) == pytest.approx(result, 0.001)
