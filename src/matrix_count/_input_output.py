# Checking input validity and hard-code checks for the combinatoric problems
from __future__ import annotations

import logging

import numpy as np

from . import _util

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _log_symmetric_matrices_check_arguments(
    row_sums,
    *,
    diagonal_sum=None,
    index_partition=None,
    block_sums=None,
    alpha=1.0,
    estimate_order=3,
    verbose=False,
):
    """Raises AssertionError for invalid inputs

    :param row_sums: Row sums of the matrix. Length n array-like of non-negative integers.
    :type row_sums: list | np.array
    :param diagonal_sum: What the sum of the diagonal elements should be constrained to.
        Either an integer greater than or equal to 0 or None, resulting in no constraint on the diagonal elements, defaults to None.
    :type diagonal_sum: int | None, optional
    :param index_partition: A list of length n of integers ranging from 1 to q.
        index_partition[i] indicates the block which index i belongs to for the purposes of a block sum constraint.
        A value of None results in no block sum constraint, defaults to None.
    :type index_partition: list of int | None, optional
    :param block_sums: A 2D (q, q) symmetric square NumPy array of non-negative integers representing the constrained sum of each block of the matrix.
        A value of None results in no block sum constraint, defaults to None.
    :type block_sums: np.ndarray, shape (q, q), dtype int
    :param alpha: Dirichlet-multinomial parameter greater than or equal to 0 to weigh the matrices in the sum.
        A value of 1 gives the uniform count of matrices, defaults to 1
    :type alpha: float, optional
    :param estimate_order: Order of moment matching estimate to use. Options: {2, 3}. Defaults to 3.
    :type estimate_order: int, optional
    :param verbose: Whether to print details of calculation. Defaults to False.
    :type verbose: bool, optional
    """

    # Checking input validity
    assert isinstance(
        row_sums, (list, np.ndarray)
    ), "row_sums must be a list or np.array"
    assert all(
        isinstance(x, (int, np.int32, np.int64)) and x >= 0 for x in row_sums
    ), "All elements in row_sums must be non-negative integers"

    if diagonal_sum is not None:
        assert isinstance(diagonal_sum, int), "diagonal_sum must be an integer"
        assert diagonal_sum >= 0, "diagonal_sum must be greater than or equal to 0"

    if index_partition is not None:
        assert isinstance(index_partition, list), "index_partition must be a list"
        assert all(
            isinstance(x, int) and x >= 1 for x in index_partition
        ), "block_indices must be a list of integers greater than or equal to 1"

        # Number of blocks
        q = np.max(index_partition)
        if block_sums is not None:
            assert isinstance(
                block_sums, np.ndarray
            ), "block_sums must be a NumPy array"
            assert block_sums.ndim == 2, "block_sums must be a 2D array"
            assert (
                block_sums.shape[0] == block_sums.shape[1]
            ), "block_sums must be a square array"
            assert block_sums.shape == (q, q), "block_sums must have shape (q, q)"
            assert block_sums.dtype == int, "block_sums must be of dtype int"
            assert np.all(
                block_sums >= 0
            ), "All elements in block_sums must be non-negative integers"
    else:
        assert (
            block_sums is None
        ), "index_partition must be provided to impose a constraint with block_sums"

    assert isinstance(alpha, (int, float)), "alpha must be a float"
    assert alpha >= 0, "alpha must be greater than or equal to 0"

    assert estimate_order in {2, 3}, "estimate_order must be either 2 or 3"

    assert isinstance(verbose, bool), "verbose must be a boolean"


def _simplify_input(
    row_sums, *, diagonal_sum=None, index_partition=None, block_sums=None
):
    # Remove instances where a row sum is 0
    row_sums = np.array(row_sums)
    if index_partition is not None:
        index_partition = index_partition[row_sums != 0]
    row_sums = row_sums[row_sums != 0]

    return row_sums, diagonal_sum, index_partition, block_sums


def _log_symmetric_matrices_hardcoded(
    row_sums,
    *,
    diagonal_sum=None,
    index_partition=None,
    block_sums=None,
    alpha=1.0,
    verbose=False,
):
    """Raises AssertionError for invalid inputs

    :param row_sums: Row sums of the matrix. Length n array-like of non-negative integers.
    :type row_sums: list | np.array
    :param diagonal_sum: What the sum of the diagonal elements should be constrained to.
        Either an integer greater than or equal to 0 or None, resulting in no constraint on the diagonal elements, defaults to None.
    :type diagonal_sum: int | None, optional
    :param index_partition: A list of length n of integers ranging from 1 to q.
        index_partition[i] indicates the block which index i belongs to for the purposes of a block sum constraint.
        A value of None results in no block sum constraint, defaults to None.
    :type index_partition: list of int | None, optional
    :param block_sums: A 2D (q, q) symmetric square NumPy array of non-negative integers representing the constrained sum of each block of the matrix.
        A value of None results in no block sum constraint, defaults to None.
    :type block_sums: np.ndarray, shape (q, q), dtype int
    :param alpha: Dirichlet-multinomial parameter greater than or equal to 0 to weigh the matrices in the sum.
        A value of 1 gives the uniform count of matrices, defaults to 1
    :type alpha: float, optional
    :param verbose: Whether to print details of calculation. Defaults to False.
    :type verbose: bool, optional
    :return: Logarithm of the number of symmetric matrices satisfying the constraints if hardcoded,
        None if no hardcoding is possible.
    :rtype: float | None
    """
    # Checking whether such a matrix is possible
    row_sums = np.array(row_sums)

    matrix_total = np.sum(row_sums)

    if matrix_total % 2 == 1:
        if verbose:
            logger.info("No matrices satisfy, margin total is odd")
        return -np.inf

    n = len(row_sums)  # Number of vertices

    if diagonal_sum is not None:
        # If the diagonal sum is constrained to be even, the provided diagonal sum must be even
        if diagonal_sum % 2 == 1:
            if verbose:
                logger.info(
                    "No matrices satisfy the even diagonal condition, given diagonal sum is odd"
                )
            return -np.inf

        # Compute the minimum possible diagonal entry that can be achieved (given the given number of off-diagonal edges)
        m_in_min = 0
        m_out = np.sum(row_sums) / 2 - diagonal_sum / 2
        for k in row_sums:
            m_in_min += max(0, np.ceil((k - m_out) / 2))

        m_in_max = np.sum(np.floor(row_sums / 2))

        if diagonal_sum < 2 * m_in_min or diagonal_sum > 2 * m_in_max:
            if verbose:
                logger.info("No matrices satisfy the diagonal sum condition.")
            return -np.inf

    if diagonal_sum == np.sum(row_sums):
        if verbose:
            logger.info("Hardcoded case: all off-diagonal entries are 0")
        return 0  # log(1)

    # TODO: Add explicit treatment of alpha = 0
    assert alpha > 0, "alpha must be greater than 0, alpha = 0 is not yet supported"

    # TODO: Add the block sums case
    assert index_partition is None, "block sum constraints are not yet supported"
    assert block_sums is None, "block sum constraints are not yet supported"

    # Explicit case where each margin is 1 (0 entries have been removed)
    if matrix_total == n:
        if verbose:
            logger.info("Hardcoded case: each margin is 1")
        # Recursively can compute that there are n!/n!! possible matrices
        return _util._log_factorial(n) - _util._log_factorial2(n)

    return None
