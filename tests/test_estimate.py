from __future__ import annotations

from matrix_count.estimate import *
import numpy as np
from pytest import approx, raises

def test_estimate_log_symmetric_matrices_invalid_arguments():
    # :param row_sums: Row sums of the matrix. Length n array-like of non-negative integers. 
    # :type row_sums: list | np.array
    with raises(AssertionError):
        estimate_log_symmetric_matrices(1)
    with raises(AssertionError):
        estimate_log_symmetric_matrices([-1, 2, 3])

    # :param diagonal_sum: What the sum of the diagonal elements should be constrained to. 
    #     Either an integer greater than or equal to 0 or None, resulting in no constraint on the diagonal elements, defaults to None.
    # :type diagonal_sum: int | None, optional
    with raises(AssertionError):
        estimate_log_symmetric_matrices([1, 2, 3], diagonal_sum=-1)

    # :param index_partition: A list of length n of integers ranging from 1 to q. 
    #     index_partition[i] indicates the block which index i belongs to for the purposes of a block sum constraint. 
    #     A value of None results in no block sum constraint, defaults to None.
    # :type index_partition: list of int | None, optional
    with raises(AssertionError):
        estimate_log_symmetric_matrices([1, 2, 3], index_partition=[1, 2, -1])

    # :param block_sums: A 2D (q, q) symmetric square NumPy array of integers representing the constrained sum of each block of the matrix. 
    #     A value of None results in no block sum constraint, defaults to None.
    # :type block_sums: np.ndarray, shape (q, q), dtype int
    with raises(AssertionError):
        estimate_log_symmetric_matrices([1, 2, 3], index_partition=[1, 2, 1], block_sums=np.array([[1, 2], [2, -1]]))
    with raises(AssertionError):
        estimate_log_symmetric_matrices([1, 2, 3], index_partition=[1, 2, 1], block_sums=np.array([[1, 2, 3], [2, 1, 3]]))

    # :param alpha: Dirichlet-multinomial parameter greater than or equal to 0 to weigh the matrices in the sum.
    #     A value of 1 gives the uniform count of matrices, defaults to 1
    # :type alpha: float, optional
    with raises(AssertionError):
        estimate_log_symmetric_matrices([1, 2, 3], alpha=-0.5)

    # :param estimate_order: Order of moment matching estimate to use. Options: {2, 3}. Defaults to 3. 
    # :type estimate_order: int, optional
    with raises(AssertionError):
        estimate_log_symmetric_matrices([1, 2, 3], estimate_order=4)

    # All good
    assert True


def test_estimate_log_symmetric_matrices_no_matrices():
    # Case: diagonal_sum is odd
    assert estimate_log_symmetric_matrices([2, 2, 2], diagonal_sum=3) == -np.inf

    # Case: diagonal_sum is greater than the sum of row_sums
    assert estimate_log_symmetric_matrices([1, 2, 3], diagonal_sum=10) == -np.inf

    # Case: max_margin - m > diagonal_sum / 2.0
    assert estimate_log_symmetric_matrices([10, 1, 1], diagonal_sum=2) == -np.inf

    # Case: block_sums is not None (not yet supported) TODO: check that this is an impossible case
    # assert estimate_log_symmetric_matrices([1, 2, 3], index_partition=[1, 2, 1], block_sums=np.array([[1, 2], [2, 1]])) == -np.inf

    # Case: total of row_sums is odd
    assert estimate_log_symmetric_matrices([1, 2, 2]) == -np.inf

    # Case: total of off-diagonal is odd
    assert estimate_log_symmetric_matrices([1, 2, 3], diagonal_sum=3) == -np.inf

    # All good
    assert True

def test_estimate_log_symmetric_matrices_hardcoded():
    # Case: each margin is 1
    assert estimate_log_symmetric_matrices([0, 1, 1, 1, 1]) == approx(np.log(3))

    # All good
    assert True

def test_alpha_2():
    # Test the second order moment matching estimates (comparing to Mathematica examples)
    assert alpha_2_symmetric_no_block(200,10,diagonal_sum=None, alpha=1.0) == approx(9.75402)

    

    # All good
    assert True

def test_alpha_3():
    # Test the second order moment matching estimates (comparing to Mathematica examples)
    alpha_pm = alpha_3_symmetric_no_block(200,10,diagonal_sum=None, alpha=1.0)
    alpha_pm_true = 13.7998, 7.53246
    assert alpha_pm[0] == approx(alpha_pm_true[0], 0.001)
    assert alpha_pm[1] == approx(alpha_pm_true[1], 0.001)
    
    # All good
    assert True