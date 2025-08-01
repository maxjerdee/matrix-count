from __future__ import annotations

import numpy as np
import pytest

from matrix_count.estimate import (
    _alpha_symmetric_2,
    _alpha_symmetric_3,
    _alpha_symmetric_binary,
    _asymmetric_block_variance,
    _symmetric_block_variance,
    _variance_to_alpha,
    _symmetric_block_log_total_weight,
    _asymmetric_block_log_total_weight,
    estimate_log_symmetric_matrices,
)

def test_estimate_log_symmetric_matrices_invalid_arguments():
    # :param row_sums: Row sums of the matrix. Length n array-like of non-negative integers.
    with pytest.raises(AssertionError):
        estimate_log_symmetric_matrices([-1, 2, 3])

    # :param diagonal_sum: What the sum of the diagonal elements should be constrained to.
    #     Either an integer greater than or equal to 0 or None, resulting in no constraint on the diagonal elements, defaults to None.
    # :type diagonal_sum: int | None, optional
    with pytest.raises(AssertionError):
        estimate_log_symmetric_matrices([1, 2, 3], diagonal_sum=-1)

    # :param index_partition: A list of length n of integers ranging from 1 to q.
    #     index_partition[i] indicates the block which index i belongs to for the purposes of a block sum constraint.
    #     A value of None results in no block sum constraint, defaults to None.
    # :type index_partition: list of int | None, optional
    with pytest.raises(AssertionError):
        estimate_log_symmetric_matrices([1, 2, 3], index_partition=[1, 2, -1])

    # :param block_sums: A 2D (q, q) symmetric square NumPy array of integers representing the constrained sum of each block of the matrix.
    #     A value of None results in no block sum constraint, defaults to None.
    # :type block_sums: np.ndarray, shape (q, q), dtype int
    with pytest.raises(AssertionError):
        estimate_log_symmetric_matrices(
            [1, 2, 3], index_partition=[1, 2, 1], block_sums=np.array([[1, 2], [2, -1]])
        )
    with pytest.raises(AssertionError):
        estimate_log_symmetric_matrices(
            [1, 2, 3],
            index_partition=[1, 2, 1],
            block_sums=np.array([[1, 2, 3], [2, 1, 3]]),
        )

    # :param alpha: Dirichlet-multinomial parameter greater than or equal to 0 to weigh the matrices in the sum.
    #     A value of 1 gives the uniform count of matrices, defaults to 1
    # :type alpha: float, optional
    with pytest.raises(AssertionError):
        estimate_log_symmetric_matrices([1, 2, 3], alpha=-0.5)

    # Case: binary_matrix margin is too large
    with pytest.raises(AssertionError):
        estimate_log_symmetric_matrices([1, 2, 3], binary_matrix=True)


def test_estimate_log_symmetric_matrices_no_matrices():
    # Case: diagonal_sum is odd
    assert estimate_log_symmetric_matrices([2, 2, 2], diagonal_sum=3) == -np.inf

    # Case: diagonal_sum is greater than the sum of row_sums
    assert estimate_log_symmetric_matrices([1, 2, 3], diagonal_sum=10) == -np.inf

    # Case: diagonal_sums with no matrices
    assert estimate_log_symmetric_matrices([10, 1, 1], diagonal_sum=2) == -np.inf
    assert estimate_log_symmetric_matrices([2, 2, 2, 2], diagonal_sum=6) == -np.inf

    # Case: block_sums is not None (not yet supported) TODO: check that this is an impossible case
    # assert estimate_log_symmetric_matrices([1, 2, 3], index_partition=[1, 2, 1], block_sums=np.array([[1, 2], [2, 1]])) == -np.inf

    # Case: total of row_sums is odd
    assert estimate_log_symmetric_matrices([1, 2, 2]) == -np.inf

    # Case: total of off-diagonal is odd
    assert estimate_log_symmetric_matrices([1, 2, 3], diagonal_sum=3) == -np.inf


def test_estimate_log_symmetric_matrices_hardcoded():
    # Case: each margin is 1
    assert estimate_log_symmetric_matrices([0, 1, 1, 1, 1]) == pytest.approx(np.log(3))

    # Case: all off-diagonal are 0
    assert estimate_log_symmetric_matrices(
        [2, 2, 2, 2], diagonal_sum=8
    ) == pytest.approx(np.log(1))

    # Case: binary_matrix margins are all n - 1
    assert estimate_log_symmetric_matrices(
        [2, 2, 2], binary_matrix=True
    ) == pytest.approx(np.log(1))

    # Check binary_matrix cases that violate the Erdos-Gallai conditions
    assert (
        estimate_log_symmetric_matrices([4, 3, 1, 1, 1], binary_matrix=True) == -np.inf
    )


def test__symmetric_block_variance():
    # binary_matrix = False, diagonal_sum = None
    assert _symmetric_block_variance(10, 3, alpha=1.5) == pytest.approx(3.8889, 0.001)
    # binary_matrix = False, diagonal_sum
    assert _symmetric_block_variance(10, 3, diagonal_sum=4, alpha=1.5) == pytest.approx(3.0101, 0.001)
    # binary_matrix = True
    assert _symmetric_block_variance(6, 4, binary_matrix=True) == pytest.approx(0.45, 0.001)


def test__asymmetric_block_variance():
    # binary_matrix = False, column_sums = None
    assert _asymmetric_block_variance(4, 3, 2, column_sums=None, alpha=1.5) == pytest.approx(1.15556, 0.001)
    # binary_matrix = False, column_sums
    assert _asymmetric_block_variance(3, 4, 2, column_sums=[2, 1], alpha=1.5) == pytest.approx(0.616071, 0.001)
    # binary_matrix = True, column_sums = None
    assert _asymmetric_block_variance(4, 3, 2, binary_matrix=True) == pytest.approx(0.355556, 0.001)
    # binary_matrix = True, column_sums
    assert _asymmetric_block_variance(3, 4, 2, column_sums=[2, 1], alpha=1.5, binary_matrix=True) == pytest.approx(0.4375, 0.001)


def test__variance_to_alpha():

    assert _variance_to_alpha(30, 200, 10) == pytest.approx(29.75, 0.001)
    assert _variance_to_alpha(10, 200, 10) == pytest.approx(-44.875, 0.001)
    # allow_pseudo = False
    assert _variance_to_alpha(10, 200, 10, allow_pseudo=False) > 100000


def test__symmetric_block_log_total_weight():
    # binary_matrix = False, diagonal_sum = None
    total = 20
    length = 3
    alpha = 1.5
    result = 10.6864
    assert _symmetric_block_log_total_weight(total, length, alpha=alpha) == pytest.approx(result, 0.001)

    # binary_matrix = False, diagonal_sum
    total = 20
    diagonal_sum = 4
    length = 3
    alpha = 1.5
    result = 8.19169
    assert _symmetric_block_log_total_weight(total, length, diagonal_sum=diagonal_sum, alpha=alpha) == pytest.approx(result, 0.001)

    # binary_matrix = True
    total = 6
    length = 4
    alpha = 1.5
    result = 4.21213
    assert _symmetric_block_log_total_weight(total, length, alpha=alpha, binary_matrix=True) == pytest.approx(result, 0.001)


def test__asymmetric_block_log_total_weight():
    # binary_matrix = False, column_sums = None
    total = 4
    length = 3
    depth = 2
    alpha = 1.5
    result = 6.20456
    assert _asymmetric_block_log_total_weight(total, length, depth, column_sums=None, alpha=alpha) == pytest.approx(result, 0.001)

    # binary_matrix = False, column_sums
    length = 4
    column_sums = [2,1]
    depth = len(column_sums)
    total = np.sum(column_sums)
    alpha = 1.5
    result = 4.83628
    assert _asymmetric_block_log_total_weight(total, length, depth, column_sums=column_sums, alpha=alpha) == pytest.approx(result, 0.001)

    # binary_matrix = True, column_sums = None
    total = 4
    length = 3
    depth = 2
    alpha = 1.5
    result = 4.32991
    assert _asymmetric_block_log_total_weight(total, length, depth, alpha=alpha, binary_matrix=True) == pytest.approx(result, 0.001)

    # binary_matrix = True, column_sums
    length = 4
    column_sums = [2,1]
    depth = len(column_sums)
    total = np.sum(column_sums)
    alpha = 1.5
    result = 4.39445
    assert _asymmetric_block_log_total_weight(total, length, depth, column_sums=column_sums, alpha=alpha, binary_matrix=True) == pytest.approx(result, 0.001)


def test__alpha_symmetric_3():
    # Test the third order moment matching estimates (comparing to Mathematica examples)
    alpha_pm = _alpha_symmetric_3(200, 10, alpha=1.0)
    alpha_pm_true = 13.7998, 7.53246
    assert alpha_pm[0] == pytest.approx(alpha_pm_true[0], 0.001)
    assert alpha_pm[1] == pytest.approx(alpha_pm_true[1], 0.001)

    alpha_pm = _alpha_symmetric_3(200, 10, alpha=5.0)
    alpha_pm_true = 63.6326, 30.4128
    assert alpha_pm[0] == pytest.approx(alpha_pm_true[0], 0.001)
    assert alpha_pm[1] == pytest.approx(alpha_pm_true[1], 0.001)


def test_estimate_log_symmetric_matrices():
    # Test the estimate_log_symmetric_matrices function
    assert estimate_log_symmetric_matrices(
        [20, 11, 3], alpha=1, force_second_order=True
    ) == pytest.approx(3.65746, 0.001)
    assert estimate_log_symmetric_matrices(
        [20, 11, 3], alpha=5, force_second_order=True
    ) == pytest.approx(20.3397, 0.001)

    assert estimate_log_symmetric_matrices([20, 11, 3], alpha=1) == pytest.approx(
        3.60119, 0.001
    )
    assert estimate_log_symmetric_matrices([20, 11, 3], alpha=5) == pytest.approx(
        20.3536, 0.001
    )

    assert estimate_log_symmetric_matrices(
        [20, 11, 3], diagonal_sum=20, alpha=1
    ) == pytest.approx(1.29499, 0.001)
    assert estimate_log_symmetric_matrices(
        [20, 11, 3], diagonal_sum=20, alpha=5
    ) == pytest.approx(18.5925, 0.001)

    # Binary matrices
    # pseudo Dirichlet-Multinomial
    assert estimate_log_symmetric_matrices(
        [1, 1, 1, 1, 2], binary_matrix=True
    ) == pytest.approx(1.89415, 0.001)

    assert estimate_log_symmetric_matrices(
        [1, 1, 1, 1, 2], alpha=5, binary_matrix=True
    ) == pytest.approx(6.72247, 0.001)

    # This is a case where direct abilication of the Dirichlet-Multinomial distribution with a negative alpha results in a negative count
    assert estimate_log_symmetric_matrices(
        [1, 1, 1, 1, 1, 1, 6], binary_matrix=True
    ) == pytest.approx(-3.01943, 0.001)

    # Multinomial estimate (allow_pseudo = False)
    assert estimate_log_symmetric_matrices(
        [1, 1, 1, 1, 2], binary_matrix=True, allow_pseudo=False
    ) == pytest.approx(1.01697, 0.001)

    assert estimate_log_symmetric_matrices(
        [1, 1, 1, 1, 2], alpha=5, binary_matrix=True, allow_pseudo=False
    ) == pytest.approx(5.84528, 0.001)
