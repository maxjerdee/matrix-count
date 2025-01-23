from __future__ import annotations

import logging

import numpy as np
from numpy.typing import ArrayLike

from . import _input_output, _util

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def alpha_symmetric_2(
    matrix_total: int, n: int, diagonal_sum: int | None = None, alpha: float = 1.0
) -> float:
    """Dirichlet-Multinomial parameter alpha for the second order moment matching estimate
        of the number of symmetric matrices with given conditions.

    :param matrix_total: Matrix total (sum of all entries).
    :type matrix_total: int
    :param n: Matrix size (n,n).
    :type n: int
    :param diagonal_sum: Sum of the diagonal elements of the matrix.
    :type diagonal_sum: int | None
    :return: alpha
    :rtype: float
    """
    if diagonal_sum is None:
        log_numerator = _util.log_sum_exp(
            [
                np.log(matrix_total),
                np.log(matrix_total + n * (matrix_total - 1) - 2)
                + np.log(n + 1)
                + np.log(alpha),
            ]
        )  # Overflow prevention

        log_denominator = _util.log_sum_exp(
            [
                np.log(2 * (matrix_total - 1)),
                np.log(n) + np.log((n + 1) * alpha + matrix_total - 2),
            ]
        )  # Overflow prevention

        return float(np.exp(log_numerator - log_denominator))
    # Fixed diagonal sum
    with np.errstate(divide="ignore"):  # Ignore divide by zero warning
        log_numerator = np.real(
            _util.log_sum_exp(
                [
                    np.log(-1 + 0j)
                    + np.log(-1 + n)
                    + 2 * np.log(diagonal_sum)
                    + np.log(2 + (-1 + n) * n * alpha),
                    np.log(-1 + 0j)
                    + np.log(2)
                    + np.log(-1 + n)
                    + np.log(n)
                    + np.log(diagonal_sum)
                    + np.log(alpha)
                    + np.log(2 + (-1 + n) * n * alpha),
                    np.log(n - 1)
                    + 2 * np.log(matrix_total)
                    + np.log(1 + n * alpha)
                    + np.log(2 + (-1 + n) * n * alpha),
                    np.log(-1 + 0j)
                    + np.log(-2 + n)
                    + np.log(matrix_total - diagonal_sum)
                    + np.log(1 + n * alpha)
                    + np.log(matrix_total - diagonal_sum + (-1 + n) * n * alpha),
                ]
            )
        )
        log_denominator = np.real(
            np.log(n)
            + _util.log_sum_exp(
                [
                    np.log(-1 + n)
                    + 2 * np.log(diagonal_sum)
                    + np.log(2 + (-1 + n) * n * alpha),
                    np.log(2)
                    + np.log(-1 + n)
                    + np.log(n)
                    + np.log(diagonal_sum)
                    + np.log(alpha)
                    + np.log(2 + (-1 + n) * n * alpha),
                    np.log(-1 + 0j)
                    + np.log(-1 + n)
                    + np.log(matrix_total)
                    + np.log(1 + n * alpha)
                    + np.log(2 + (-1 + n) * n * alpha),
                    np.log(-2 + n)
                    + np.log(matrix_total - diagonal_sum)
                    + np.log(1 + n * alpha)
                    + np.log(matrix_total - diagonal_sum + (-1 + n) * n * alpha),
                ]
            )
        )
        return float(np.exp(log_numerator - log_denominator))


def alpha_symmetric_3(
    matrix_total: int, n: int, diagonal_sum: int | None = None, alpha: float = 1.0
) -> tuple[float, float]:
    """Dirichlet-Multinomial parameters alpha_plus and alpha_minus for the third order moment matching estimate
        of the number of symmetric matrices with given conditions.

    :param matrix_total: Matrix total (sum of all entries).
    :type matrix_total: int
    :param n: Matrix size (n,n).
    :type n: int
    :param diagonal_sum: Sum of the diagonal elements of the matrix.
    :type diagonal_sum: int | None
    :return: alpha_plus, alpha_minus
    :rtype: float, float
    """
    if diagonal_sum is None:
        log_common_numerator = np.real(
            _util.log_sum_exp(
                [
                    2 * _util.log_c(matrix_total)
                    + _util.log_c(
                        4
                        + n
                        + (1 + n) * (4 + n * (11 + 4 * n)) * alpha
                        + n * (1 + n) ** 3 * (2 + n) * alpha**2
                    ),
                    _util.log_c(1 + n)
                    + _util.log_c(matrix_total)
                    + _util.log_c(
                        -4
                        + alpha
                        * (
                            -12
                            + n
                            * (
                                -17
                                - 11 * n
                                - (1 + n) * (7 + n * (4 + 3 * n)) * alpha
                                + n * (1 + n) ** 3 * alpha**2
                            )
                        )
                    ),
                    _util.log_c(1 + n)
                    + 2 * _util.log_c(alpha)
                    + _util.log_c(
                        8
                        + n
                        * (
                            4
                            + alpha * (2 + n * (7 + 2 * n - (1 + n) * (4 + n) * alpha))
                        )
                    ),
                ]
            )
        )

        log_sqrt_term = np.real(
            0.5
            * (
                _util.log_c(-1 + alpha + n * alpha)
                + _util.log_c(matrix_total - (1 + n) * alpha)
                + _util.log_c(matrix_total + n * (1 + n) * alpha)
                + _util.log_c(
                    -4
                    - 4 * n
                    + 4 * matrix_total
                    + 3 * n * matrix_total
                    + n * (1 + n) * (n * (-1 + matrix_total) + matrix_total) * alpha
                )
                + _util.log_c(
                    4 * (-1 + matrix_total)
                    + n
                    * (
                        -4
                        + 5 * matrix_total
                        + (1 + n)
                        * (-5 + 4 * matrix_total + n * (-4 + 5 * matrix_total))
                        * alpha
                        + n
                        * (1 + n) ** 2
                        * (-2 + n * (-1 + matrix_total) + matrix_total)
                        * alpha**2
                    )
                )
            )
        )

        log_denominator = np.real(
            _util.log_sum_exp(
                [
                    _util.log_c(8 * (1 + n) ** 2),
                    _util.log_c(-2)
                    + _util.log_c(1 + n)
                    + _util.log_c(8 + 5 * n)
                    + _util.log_c(matrix_total),
                    _util.log_c(2 * (4 + n * (5 + 2 * n)))
                    + 2 * _util.log_c(matrix_total),
                    _util.log_c(n * (1 + n) * alpha)
                    + _util.log_c(
                        2 * n * (1 + 2 * n)
                        - 3 * n * (3 + n) * matrix_total
                        + (4 + n * (3 + n)) * matrix_total**2
                        - 2 * (1 + matrix_total)
                    ),
                    2 * _util.log_c(n * (1 + n) * alpha)
                    + _util.log_c(-3 + n * (-5 + matrix_total) + 5 * matrix_total),
                    _util.log_c(2)
                    + 3 * (_util.log_c(n) + _util.log_c(1 + n) + _util.log_c(alpha)),
                ]
            )
        )

        alpha_plus = np.real(
            np.exp(
                _util.log_sum_exp([log_common_numerator, log_sqrt_term])
                - log_denominator
            )
        )
        alpha_minus = np.real(
            np.exp(
                _util.log_sum_exp(
                    [log_common_numerator, np.log(-1 + 0j) + log_sqrt_term]
                )
                - log_denominator
            )
        )

        return alpha_plus, alpha_minus
    # Fixed diagonal sum
    raise NotImplementedError


def estimate_log_symmetric_matrices(
    row_sums: list[int] | ArrayLike,
    *,
    diagonal_sum: int | None = None,
    index_partition: list[int] | None = None,
    block_sums: ArrayLike | None = None,
    alpha: float = 1.0,
    estimate_order: int = 3,
    verbose: bool = False,
) -> float:
    """Dirichlet-multinomial moment-matching estimate of the logarithm
        of the number of symmetric non-negative matrices with given row sums.

    :param row_sums: Row sums of the matrix. Length n array-like of non-negative integers.
    :type row_sums: list | np.array
    :param diagonal_sum: What the sum of the diagonal elements should be constrained to.
        Either an integer greater than or equal to 0 or None, resulting in no constraint on the diagonal elements, defaults to None.
    :type diagonal_sum: int | None, optional
    :param index_partition: A list of length n of integers ranging from 1 to q.
        index_partition[i] indicates the block which index i belongs to for the purposes of a block sum constraint.
        A value of None results in no block sum constraint, defaults to None.
    :type index_partition: list of int | None, optional
    :param block_sums: A 2D (q, q) symmetric square NumPy array of integers representing the constrained sum of each block of the matrix.
        A value of None results in no block sum constraint, defaults to None.
    :type block_sums: np.ndarray, shape (q, q), dtype int
    :param alpha: Dirichlet-multinomial parameter greater than or equal to 0 to weigh the matrices in the sum.
        A value of 1 gives the uniform count of matrices, defaults to 1
    :type alpha: float, optional
    :param estimate_order: Order of moment matching estimate to use. Options: {2, 3}. Defaults to 3.
    :type estimate_order: int, optional
    :param verbose: Whether to print details of calculation. Defaults to False.
    :type verbose: bool, optional
    :return: The logarithm of the estimate of the number of symmetric matrices with given row sums and conditions
    :rtype: float
    """
    # Check input validity
    _input_output.log_symmetric_matrices_check_arguments(
        row_sums,
        diagonal_sum=diagonal_sum,
        index_partition=index_partition,
        block_sums=block_sums,
        alpha=alpha,
        estimate_order=estimate_order,
        verbose=verbose,
    )

    # Remove empty margins
    row_sums, diagonal_sum, index_partition, block_sums = _input_output.simplify_input(
        row_sums,
        diagonal_sum=diagonal_sum,
        index_partition=index_partition,
        block_sums=block_sums,
    )

    # Check for hardcoded cases
    hardcoded_result = _input_output.log_symmetric_matrices_hardcoded(
        row_sums,
        diagonal_sum=diagonal_sum,
        index_partition=index_partition,
        block_sums=block_sums,
        alpha=alpha,
        verbose=verbose,
    )
    if hardcoded_result is not None:
        return hardcoded_result

    matrix_total: int = np.sum(row_sums)
    n = len(row_sums)
    if diagonal_sum is None:
        if estimate_order == 2:
            alpha_dm = alpha_symmetric_2(
                matrix_total, n, diagonal_sum=diagonal_sum, alpha=alpha
            )
            result = _util.log_binom(
                matrix_total / 2 + alpha * n * (n + 1) / 2 - 1,
                alpha * n * (n + 1) / 2 - 1,
            )
            log_p = -_util.log_binom(matrix_total + n * alpha_dm - 1, n * alpha_dm - 1)
            for k in row_sums:
                log_p += _util.log_binom(k + alpha_dm - 1, alpha_dm - 1)
            result += log_p
            return result
        if estimate_order == 3:
            alpha_plus, alpha_minus = alpha_symmetric_3(
                matrix_total, n, diagonal_sum=diagonal_sum, alpha=alpha
            )
            log_1 = _util.log_binom(
                matrix_total / 2 + alpha * n * (n + 1) / 2 - 1,
                alpha * n * (n + 1) / 2 - 1,
            )
            log_1 += -_util.log_binom(
                matrix_total + n * alpha_plus - 1, n * alpha_plus - 1
            )
            for k in row_sums:
                log_1 += _util.log_binom(k + alpha_plus - 1, alpha_plus - 1)
            log_2 = _util.log_binom(
                matrix_total / 2 + alpha * n * (n + 1) / 2 - 1,
                alpha * n * (n + 1) / 2 - 1,
            )
            log_2 += -_util.log_binom(
                matrix_total + n * alpha_minus - 1, n * alpha_minus - 1
            )
            for k in row_sums:
                log_2 += _util.log_binom(k + alpha_minus - 1, alpha_minus - 1)
            return _util.log_sum_exp([log_1, log_2]) - float(np.log(2))
        raise NotImplementedError

    if estimate_order == 3:
        if verbose:
            logger.info(
                "3rd order estimate of symmetric matrices with fixed diagonal count not yet implemented. Defaulting to 2nd order estimate."
            )
        estimate_order = 2

    if estimate_order == 2:
        alpha_dm = alpha_symmetric_2(
            matrix_total, n, diagonal_sum=diagonal_sum, alpha=alpha
        )
        result = _util.log_binom(diagonal_sum / 2 + alpha * n - 1, alpha * n - 1)
        result += _util.log_binom(
            (matrix_total - diagonal_sum) / 2 + alpha * n * (n - 1) / 2 - 1,
            alpha * n * (n - 1) / 2 - 1,
        )
        result += -_util.log_binom(matrix_total + n * alpha_dm - 1, n * alpha_dm - 1)
        for k in row_sums:
            result += _util.log_binom(k + alpha_dm - 1, alpha_dm - 1)
        return result

    raise NotImplementedError  # estimate_order != 2, 3
