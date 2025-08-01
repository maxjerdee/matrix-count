from __future__ import annotations

import logging

import numpy as np
from numpy.typing import ArrayLike

from . import _input_output, _util

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _alpha_symmetric_2(
    total: int, n: int, diagonal_sum: int | None = None, alpha: float = 1.0
) -> float:
    """
    Dirichlet-Multinomial parameter alpha for the second order moment matching estimate
    of the number of symmetric matrices with given conditions.

    Parameters
    ----------
    total : int
        Matrix total (sum of all entries).
    n : int
        Matrix size (n, n).
    diagonal_sum : int or None, optional
        Sum of the diagonal elements of the matrix.
    alpha : float, optional
        Dirichlet-multinomial parameter, defaults to 1.0.

    Returns
    -------
    float
        alpha
    """
    if diagonal_sum is None:
        log_numerator = _util._log_sum_exp(
            [
                np.log(total),
                np.log(total + n * (total - 1) - 2)
                + np.log(n + 1)
                + np.log(alpha),
            ]
        )  # Overflow prevention

        log_denominator = _util._log_sum_exp(
            [
                np.log(2 * (total - 1)),
                np.log(n) + np.log((n + 1) * alpha + total - 2),
            ]
        )  # Overflow prevention

        return float(np.exp(log_numerator - log_denominator))
    # Fixed diagonal sum
    with np.errstate(divide="ignore"):  # Ignore divide by zero warning
        log_numerator = np.real(
            _util._log_sum_exp(
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
                    + 2 * np.log(total)
                    + np.log(1 + n * alpha)
                    + np.log(2 + (-1 + n) * n * alpha),
                    np.log(-1 + 0j)
                    + np.log(-2 + n)
                    + np.log(total - diagonal_sum)
                    + np.log(1 + n * alpha)
                    + np.log(total - diagonal_sum + (-1 + n) * n * alpha),
                ]
            )
        )
        log_denominator = np.real(
            np.log(n)
            + _util._log_sum_exp(
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
                    + np.log(total)
                    + np.log(1 + n * alpha)
                    + np.log(2 + (-1 + n) * n * alpha),
                    np.log(-2 + n)
                    + np.log(total - diagonal_sum)
                    + np.log(1 + n * alpha)
                    + np.log(total - diagonal_sum + (-1 + n) * n * alpha),
                ]
            )
        )
        return float(np.exp(log_numerator - log_denominator))


def _alpha_symmetric_3(
    total: int, n: int, alpha: float = 1.0
) -> tuple[float, float]:
    """
    Dirichlet-Multinomial parameters alpha_plus and alpha_minus for the third order moment matching estimate
    of the number of symmetric matrices.

    Parameters
    ----------
    total : int
        Matrix total (sum of all entries).
    n : int
        Matrix size (n, n).
    alpha : float, optional
        Dirichlet-multinomial parameter, defaults to 1.0.

    Returns
    -------
    tuple of (float, float)
        alpha_plus, alpha_minus
    """
    log_common_numerator = np.real(
        _util._log_sum_exp(
            [
                2 * _util._log_c(total)
                + _util._log_c(
                    4
                    + n
                    + (1 + n) * (4 + n * (11 + 4 * n)) * alpha
                    + n * (1 + n) ** 3 * (2 + n) * alpha**2
                ),
                _util._log_c(1 + n)
                + _util._log_c(total)
                + _util._log_c(
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
                _util._log_c(1 + n)
                + 2 * _util._log_c(alpha)
                + _util._log_c(
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
            _util._log_c(-1 + alpha + n * alpha)
            + _util._log_c(total - (1 + n) * alpha)
            + _util._log_c(total + n * (1 + n) * alpha)
            + _util._log_c(
                -4
                - 4 * n
                + 4 * total
                + 3 * n * total
                + n * (1 + n) * (n * (-1 + total) + total) * alpha
            )
            + _util._log_c(
                4 * (-1 + total)
                + n
                * (
                    -4
                    + 5 * total
                    + (1 + n)
                    * (-5 + 4 * total + n * (-4 + 5 * total))
                    * alpha
                    + n
                    * (1 + n) ** 2
                    * (-2 + n * (-1 + total) + total)
                    * alpha**2
                )
            )
        )
    )

    log_denominator = np.real(
        _util._log_sum_exp(
            [
                _util._log_c(8 * (1 + n) ** 2),
                _util._log_c(-2)
                + _util._log_c(1 + n)
                + _util._log_c(8 + 5 * n)
                + _util._log_c(total),
                _util._log_c(2 * (4 + n * (5 + 2 * n)))
                + 2 * _util._log_c(total),
                _util._log_c(n * (1 + n) * alpha)
                + _util._log_c(
                    2 * n * (1 + 2 * n)
                    - 3 * n * (3 + n) * total
                    + (4 + n * (3 + n)) * total**2
                    - 2 * (1 + total)
                ),
                2 * _util._log_c(n * (1 + n) * alpha)
                + _util._log_c(-3 + n * (-5 + total) + 5 * total),
                _util._log_c(2)
                + 3 * (_util._log_c(n) + _util._log_c(1 + n) + _util._log_c(alpha)),
            ]
        )
    )

    alpha_plus = np.real(
        np.exp(
            _util._log_sum_exp([log_common_numerator, log_sqrt_term])
            - log_denominator
        )
    )
    alpha_minus = np.real(
        np.exp(
            _util._log_sum_exp(
                [log_common_numerator, np.log(-1 + 0j) + log_sqrt_term]
            )
            - log_denominator
        )
    )

    return alpha_plus, alpha_minus


def _alpha_symmetric_binary(total: int, n: int) -> float:
    """
    Dirichlet-Multinomial parameter alpha for the second order moment matching estimate
    of the number of binary symmetric matrices. Note that this will typically be negative, and not a valid Dirichlet-Multinomial parameter.

    Parameters
    ----------
    total : int
        Matrix total (sum of all entries).
    n : int
        Matrix size (n, n).

    Returns
    -------
    float
        alpha
    """
    alpha_epsilon = 1e-10  # To avoid division by zero
    return (-total * n + n - 1) / (total + n - 1 + alpha_epsilon)

def _symmetric_block_variance(total: int, length: int, *, diagonal_sum: int | None = None, alpha: float = 1.0, binary_matrix: bool = False) -> float:
    """
    Variance of each row sum entry for a symmetric length x length block with given total and alpha Dirichlet-multinomial weight.

    Parameters
    ----------
    total : int
        Total sum of the block. Must be even.
    length : int
        Block is length x length.
    diagonal_sum : int or None, optional
        Sum of the diagonal elements. If None, no constraint is applied.
    alpha : float, optional
        Dirichlet-multinomial parameter for weighting the matrices in the sum. Defaults to 1.0.
    binary_matrix : bool, optional
        Whether the matrix is binary (0 or 1) instead of non-negative integer valued.

    Returns
    -------
    float
        Variance of each row sum entry
    """
    assert total % 2 == 0, "Total must be even for symmetric matrices."

    if not binary_matrix: # Non-negative integer matrix
        if diagonal_sum is None: # Unconstrained diagonal
            return ((-2 + length + length**2) * total * (alpha * length * (1 + length) + total)) / (
                length**2 * (1 + length) * (2 + alpha * length * (1 + length)))
        else: # Constrained diagonal
            return ((diagonal_sum * (alpha * (-1 + length) * length * (6 + length * (-1 + alpha * length)) + diagonal_sum * (-4 + 3 * length + alpha * length * (-1 + (-1 + length) * length))) + (-2 + length) * (1 + alpha * length) * (-2 * diagonal_sum + alpha * (-1 + length) * length) * total + (-2 + length) * (1 + alpha * length) * total**2)) / (length**2 * (1 + alpha * length) * (2 + alpha * (-1 + length) * length))
    else:  # Binary matrix
        return -((total * (total + length - length**2)) / (length**2 * (1 + length)))  # Binary matrix variance formula

def _asymmetric_block_variance(total: int, length: int, depth: int, *, column_sums: list[int] | None = None, alpha: float = 1.0, binary_matrix: bool = False) -> float:
    """
    Variance of each row sum entry for an asymmetric length x depth block with given total and alpha Dirichlet-multinomial weight.

    Parameters
    ----------
    total : int
        Total sum of the block.
    length : int
        Length of the block.
    depth : int
        Depth of the block.
    column_sums : list[int] or None, optional
        Sums of the columns. If None, no constraint is applied.
    alpha : float, optional
        Dirichlet-multinomial parameter for weighting the matrices in the sum. Defaults to 1.0.
    binary_matrix : bool, optional
        Whether the matrix is binary (0 or 1) instead of non-negative integer valued.

    Returns
    -------
    float
        Variance of each row sum entry.
    """

    if column_sums is not None: # Column sums provided
        column_sums_squared = np.sum(np.square(column_sums))
        if not binary_matrix:
            return ((-1 + length) * (column_sums_squared + alpha * length * total)) / (length**2 * (1 + alpha * length))
        else:  # Binary matrix
            return ((-column_sums_squared + length * total) / length**2)
    else: # Column sums not provided
        if not binary_matrix:
            return ((-1 + length) * total * (alpha * depth * length + total)) / (length**2 * (1 + alpha * depth * length))
        else:  # Binary matrix
            return (((-1 + length) * (depth * length - total) * total) / (length**2 * (-1 + depth * length)))
            

def _variance_to_alpha(variance: float, total: int, length: int, allow_pseudo: bool = True, verbose: bool = False) -> float:
    """
    Find the parameter alpha for which the variance of the entries of a Dirichlet-Multinomial distribution with given total and length is equal to the specified variance.

    Parameters
    ----------
    variance : float
        Variance to be matched.
    total : int
        Total sum of the vector.
    length : int
        Length of the vector.
    allow_pseudo : bool, optional
        Whether to allow the use of negative alpha values when moment matching. Defaults to True.
    verbose : bool, optional
        Whether to print details of calculation. Defaults to False.
        
    Returns
    -------
    float
        Dirichlet-Multinomial parameter alpha.
    """
    numerator = (-1 + length) * total**2 - length**2 * variance
    denominator = length * (total - length * total + length**2 * variance)
    if denominator == 0:
        if verbose:
            logger.warning(
                "Denominator is zero, returning infinity for alpha."
            )
        return float('inf')  # Avoid division by zero
    else:
        alpha = numerator / denominator

    if not allow_pseudo and alpha < 0:
        if verbose:
            logger.warning(
                "Moment-matching alpha is negative, which is not a valid Dirichlet-Multinomial parameter. Returning infinity."
            )
        # alpha = inf is the closest we can get to matching the variance
        return float('inf')

    return alpha

def _symmetric_block_log_total_weight(total: int, length: int, *, diagonal_sum: int | None = None, alpha: float = 1.0, binary_matrix: bool = False) -> float:
    """
    Total weight over symmetric length x length blocks with given total and alpha Dirichlet-multinomial weight.

    Parameters
    ----------
    total : int
        Total sum of the block. Must be even.
    length : int
        Block is length x length.
    diagonal_sum : int or None, optional
        Sum of the diagonal elements. If None, no constraint is applied.
    alpha : float, optional
        Dirichlet-multinomial parameter for weighting the matrices in the sum. Defaults to 1.0.
    binary_matrix : bool, optional
        Whether the matrix is binary (0 or 1) instead of non-negative integer valued.

    Returns
    -------
    float
        Variance of each row sum entry
    """
    assert total % 2 == 0, "Total must be even for symmetric matrices."

    if binary_matrix:
        return _util._log_binom(length * (length - 1) / 2, total / 2) + total / 2 * np.log(alpha)    
    else:
        if diagonal_sum is None:
            return _util._log_binom(
                total / 2 + alpha * length * (length + 1) / 2 - 1,
                alpha * length * (length + 1) / 2 - 1,
            )
        else:
            return _util._log_binom(
                diagonal_sum / 2 + alpha * length - 1, alpha * length - 1
            ) + _util._log_binom(
                (total - diagonal_sum) / 2 + alpha * length * (length - 1) / 2 - 1,
                alpha * length * (length - 1) / 2 - 1,
            )
    
def _asymmetric_block_log_total_weight(total: int, length: int, depth: int, *, column_sums: list[int] | None = None, alpha: float = 1.0, binary_matrix: bool = False) -> float:
    """
    Total weight over asymmetric length x depth blocks with given total and alpha Dirichlet-multinomial weight.

    Parameters
    ----------
    total : int
        Total sum of the block.
    length : int
        Length of the block.
    depth : int
        Depth of the block.
    column_sums : list[int] or None, optional
        Sums of the columns. If None, no constraint is applied.
    alpha : float, optional
        Dirichlet-multinomial parameter for weighting the matrices in the sum. Defaults to 1.0.
    binary_matrix : bool, optional
        Whether the matrix is binary (0 or 1) instead of non-negative integer valued.

    Returns
    -------
    float
        Variance of each row sum entry.
    """

    if column_sums is not None: # Column sums provided
        if not binary_matrix:
            log_total = 0
            for column_sum in column_sums:
                log_total += _util._log_binom(column_sum + length * alpha - 1, length * alpha - 1)
            return log_total
        else:  # Binary matrix
            log_total = 0
            for column_sum in column_sums:
                log_total += _util._log_binom(length, column_sum) + column_sum * np.log(alpha)
            return log_total
    else: # Column sums not provided
        if not binary_matrix:
            return _util._log_binom(total + length * depth * alpha - 1, length * depth * alpha - 1)
        else:  # Binary matrix
            return _util._log_binom(length * depth, total) + total * np.log(alpha)
           

def estimate_log_symmetric_matrices(
    row_sums: list[int] | ArrayLike,
    *,
    diagonal_sum: int | None = None,
    index_partition: list[int] | None = None,
    block_sums: ArrayLike | None = None,
    block_diagonal_sums: ArrayLike | None = None,
    alpha: float = 1.0,
    force_second_order: bool = False,
    binary_matrix: bool = False,
    allow_pseudo: bool = True,
    verbose: bool = False,
) -> float:
    """
    Dirichlet-multinomial moment-matching estimate of the logarithm
    of the number of symmetric matrices with given row sums.

    Parameters
    ----------
    row_sums : ArrayLike
        Row sums of the matrix. Length n array-like of non-negative integers.
    diagonal_sum : int or None, optional
        Sum of the diagonal elements of the matrix, defaults to None, no constraint.
    index_partition : list of int or None, optional
        A list of length n of integers ranging from 1 to q.
        index_partition[i] indicates the block which index i belongs to for the purposes of a block sum constraint.
        A value of None results in no block sum constraint, defaults to None.
    block_sums : ArrayLike, optional
        A 2D (q, q) symmetric square NumPy array of integers representing the constrained sum of each block of the matrix.
        A value of None results in no block sum constraint, defaults to None.
    block_diagonal_sums : ArrayLike, optional
        A length q vector of integers representing the constrained sums of the diagonal elements of each block, defaults to None, no constraint.
    alpha : float, optional
        Dirichlet-multinomial parameter greater than or equal to 0 to weigh the matrices in the sum.
        A value of 1 gives the uniform count of matrices, defaults to 1.
    force_second_order : bool, optional
        Whether to force the use of the second order estimate. Defaults to False.
    binary_matrix : bool, optional
        Whether the matrix is binary (0 or 1) instead of non-negative integer valued. Defaults to False.
    allow_pseudo : bool, optional
        Whether to allow the use of negative alpha values when moment matching. Defaults to True.
    verbose : bool, optional
        Whether to print details of calculation. Defaults to False.

    Returns
    -------
    float
        The logarithm of the estimate of the number of symmetric matrices with given row sums and conditions.
    """
    # Check input validity
    _input_output._log_symmetric_matrices_check_arguments(
        row_sums,
        binary_matrix=binary_matrix,
        diagonal_sum=diagonal_sum,
        index_partition=index_partition,
        block_sums=block_sums,
        block_diagonal_sums=block_diagonal_sums,
        alpha=alpha,
        force_second_order=force_second_order,
        verbose=verbose,
    )

    # Remove empty margins
    row_sums, diagonal_sum, index_partition = _input_output._symmetric_simplify_input(
        row_sums,
        binary_matrix=binary_matrix,
        diagonal_sum=diagonal_sum,
        index_partition=index_partition,
        verbose=verbose,
    )

    # Check for hardcoded cases
    hardcoded_result = _input_output._log_symmetric_matrices_hardcoded(
        row_sums,
        binary_matrix=binary_matrix,
        diagonal_sum=diagonal_sum,
        index_partition=index_partition,
        block_sums=block_sums,
        block_diagonal_sums=block_diagonal_sums,
        alpha=alpha,
        verbose=verbose,
    )

    if hardcoded_result is not None:
        return hardcoded_result

    total: int = np.sum(row_sums)
    length = len(row_sums)
    # Treat the case where the matrix is non-negative, no diagonal constraint, and no block sums separately
    # since we can use the third order moment matching estimate
    if not binary_matrix and diagonal_sum is None and block_sums is None and index_partition is None and not force_second_order:
        alpha_plus, alpha_minus = _alpha_symmetric_3(
            total, length, alpha=alpha
        )
        log_count_all = _util._log_binom(
            total / 2 + alpha * length * (length + 1) / 2 - 1,
            alpha * length * (length + 1) / 2 - 1,
        )
        log_P_1 = _util._log_P_dirichlet_multinomial(
            row_sums, alpha_plus
        )
        log_P_2 = _util._log_P_dirichlet_multinomial(
            row_sums, alpha_minus
        )
        return log_count_all + _util._log_sum_exp([log_P_1, log_P_2]) - float(np.log(2)) # Mean of the probabilities
    # For all other cases we can use the second order moment matching estimate and the same general format.
    # Compute the log size of the larger superset of matrices
    if block_sums is None:
        log_count_all = _symmetric_block_log_total_weight(
            total,
            length,
            diagonal_sum=diagonal_sum,
            alpha=alpha,
            binary_matrix=binary_matrix,
        )
    else: # Block sums are present, sizes of the groups are given by the index_partition
        num_blocks = block_sums.shape[0]
        group_sizes = np.zeros(num_blocks, dtype=int)
        for block in index_partition:
            group_sizes[block - 1] += 1
        log_count_all = 0
        for r in range(num_blocks):
            if block_diagonal_sums is not None:
                diagonal_sum = block_diagonal_sums[r]
            else:
                diagonal_sum = None
            log_count_all += _symmetric_block_log_total_weight(
                block_sums[r, r],
                group_sizes[r],
                diagonal_sum=diagonal_sum,
                alpha=alpha,
                binary_matrix=binary_matrix,
            )
            for c in range(r + 1, num_blocks):
                log_count_all += _asymmetric_block_log_total_weight(
                    block_sums[r, c],
                    group_sizes[r],
                    group_sizes[c],
                    column_sums=None,
                    alpha=alpha,
                    binary_matrix=binary_matrix,
                )

    # Compute the variance of the row sums under the superset distribution
    if block_sums is None:
        variance = _symmetric_block_variance(
            total, length, diagonal_sum=diagonal_sum, alpha=alpha, binary_matrix=binary_matrix
        )
    else: # Block sum variances add (independent blocks). Can be different for each block
        variances = []
        for r in range(num_blocks):
            variance = 0
            if block_diagonal_sums is not None:
                diagonal_sum = block_diagonal_sums[r]
            else:
                diagonal_sum = None
            for c in range(num_blocks):
                if c == r:
                    variance += _symmetric_block_variance(
                        block_sums[r, r],
                        group_sizes[r],
                        diagonal_sum=diagonal_sum,
                        alpha=alpha,
                        binary_matrix=binary_matrix,
                    )
                else:
                    variance += _asymmetric_block_variance(
                        block_sums[r, c],
                        group_sizes[r],
                        group_sizes[c],
                        column_sums=None,
                        alpha=alpha,
                        binary_matrix=binary_matrix,
                    )
            variances.append(variance)

    # Compute the (log) probability of observing the desired margin under the moment-matched distribution.
    if block_sums is None:
        alpha_matched = _variance_to_alpha(
            variance,
            total,
            length,
            allow_pseudo=allow_pseudo,
            verbose=verbose,
        )
        log_P = _util._log_P_dirichlet_multinomial(
            row_sums, alpha_matched
        )
        print(row_sums, alpha_matched)
    else:
        log_P = 0
        for r in range(num_blocks):
            alpha_matched = _variance_to_alpha(variances[r])
            row_sums_in_block = []
            for i, index in enumerate(index_partition):
                if index - 1 == r:
                    row_sums_in_block.append(row_sums[i])
            log_P += _util._log_P_dirichlet_multinomial(
                row_sums_in_block, alpha_matched
            )
    if verbose:
        print(f"Log count (all blocks): {log_count_all}")
        print(f"Log probability (all blocks): {log_P}")
    print(f"row sums: {row_sums}, binary_matrix: {binary_matrix}, variance: {variance}, alpha_matched: {alpha_matched}, log_P: {log_P}, log_count_all: {log_count_all}")
    # Return the log count of the matrices
    return log_count_all + log_P

    # if not binary_matrix:  # Symmetric, non-negative matrices
    #     if diagonal_sum is None:  # Symmetric, non-negative matrices, no diagonal constraint
    #         if force_second_order:  # Only case where we might force use of the second order estimate
    #             alpha_dm = _alpha_symmetric_2(
    #                 total, n, diagonal_sum=diagonal_sum, alpha=alpha
    #             )
    #             alpha_dm = _variance_to_alpha(
    #                 _symmetric_block_variance(total, n, diagonal_sum=diagonal_sum, alpha=alpha, binary_matrix=binary_matrix),
    #             log_count_all = _util._log_binom(
    #                 total / 2 + alpha * n * (n + 1) / 2 - 1,
    #                 alpha * n * (n + 1) / 2 - 1,
    #             )
    #             log_P = _util._log_P_dirichlet_multinomial(
    #                 row_sums, alpha_dm
    #             )
    #             return log_count_all + log_P
    #         alpha_plus, alpha_minus = _alpha_symmetric_3(
    #             total, n, diagonal_sum=diagonal_sum, alpha=alpha
    #         )
    #         log_count_all = _util._log_binom(
    #             total / 2 + alpha * n * (n + 1) / 2 - 1,
    #             alpha * n * (n + 1) / 2 - 1,
    #         )
    #         log_P_1 = _util._log_P_dirichlet_multinomial(
    #             row_sums, alpha_plus
    #         )
    #         log_P_2 = _util._log_P_dirichlet_multinomial(
    #             row_sums, alpha_minus
    #         )
    #         return log_count_all + _util._log_sum_exp([log_P_1, log_P_2]) - float(np.log(2)) # Mean of the probabilities
    #     # Symmetric, non-negative matrices, diagonal constraint
    #     alpha_dm = _alpha_symmetric_2(
    #         total, n, diagonal_sum=diagonal_sum, alpha=alpha
    #     )
    #     log_count_all = _util._log_binom(diagonal_sum / 2 + alpha * n - 1, alpha * n - 1)
    #     log_count_all += _util._log_binom(
    #         (total - diagonal_sum) / 2 + alpha * n * (n - 1) / 2 - 1,
    #         alpha * n * (n - 1) / 2 - 1,
    #     )
    #     log_P = _util._log_P_dirichlet_multinomial(
    #         row_sums, alpha_dm
    #     )
    #     return log_count_all + log_P
    # else:  # Binary symmetric matrices
    #     if binary_multinomial_estimate:  # Use the multinomial estimate for binary matrices (advantage of never being too far off)
    #         result = float(
    #             _util._log_binom(n * (n - 1) / 2, total / 2)
    #             - total * np.log(n)
    #             + _util._log_factorial(total)
    #         )
    #         for k in row_sums:
    #             result -= _util._log_factorial(k)
    #         # Dirichlet-multinomial weight (note that this is a trivial calculation for binary matrices)
    #         result += total / 2 * np.log(alpha)
    #         return result
    #     else: # Binary symmetric matrices, pseudo Dirichlet-Multinomial estimate
    #         alpha_dm = _alpha_symmetric_binary(total, n) # Can be negative
    #         log_count_all = _util._log_binom(n * (n - 1) / 2, total / 2) + total / 2 * np.log(alpha)
    #         log_P = _util._log_P_dirichlet_multinomial(
    #             row_sums, alpha_dm
    #         )
    #         return log_count_all + log_P


# def estimate_log_asymmetric_matrices(
#     row_sums: list[int] | ArrayLike,
#     column_sums: list[int] | ArrayLike,
#     *,
#     alpha: float = 1.0,
#     binary_matrix: bool = False,
#     binary_multinomial_estimate: bool = False,
#     use_short_dimension: bool = True,
#     verbose: bool = False,
# ) -> float:
#     """
#     Dirichlet-multinomial moment-matching estimate of the logarithm
#     of the number of asymmetric matrices with given row and column sums.

#     Parameters
#     ----------
#     row_sums : ArrayLike
#         Row sums of the matrix. Length n array-like of non-negative integers.
#     column_sums : ArrayLike
#         Column sums of the matrix. Length m array-like of non-negative integers.
#     alpha : float, optional
#         Dirichlet-multinomial parameter greater than or equal to 0 to weigh the matrices in the sum.
#         A value of 1 gives the uniform count of matrices, defaults to 1.
#     binary_matrix : bool, optional
#         Whether the matrix is binary (0 or 1) instead of non-negative integer valued. Defaults to False.
#     binary_multinomial_estimate : bool, optional
#         Whether to use the Multinomial estimate for binary matrices instead of the pseudo Dirichlet-Multinomial estimate. Defaults to False.
#     use_short_dimension : bool, optional
#         Whether to define the shorter dimension as the rows to improve performance. Estimate is otherwise asymmetric. Defaults to True.
#     verbose : bool, optional
#         Whether to print details of calculation. Defaults to False.

#     Returns
#     -------
#     float
#         The logarithm of the estimate of the number of symmetric matrices with given row sums and conditions.
#     """
#     # Check input validity
#     _input_output._log_symmetric_matrices_check_arguments(
#         row_sums,
#         binary_matrix=binary_matrix,
#         diagonal_sum=diagonal_sum,
#         index_partition=index_partition,
#         block_sums=block_sums,
#         alpha=alpha,
#         force_second_order=force_second_order,
#         verbose=verbose,
#     )

#     # Remove empty margins
#     row_sums, diagonal_sum, index_partition, block_sums = _input_output._symmetric_simplify_input(
#         row_sums,
#         binary_matrix=binary_matrix,
#         diagonal_sum=diagonal_sum,
#         index_partition=index_partition,
#         block_sums=block_sums,
#         verbose=verbose,
#     )

#     # Check for hardcoded cases
#     hardcoded_result = _input_output._log_symmetric_matrices_hardcoded(
#         row_sums,
#         binary_matrix=binary_matrix,
#         diagonal_sum=diagonal_sum,
#         index_partition=index_partition,
#         block_sums=block_sums,
#         alpha=alpha,
#         verbose=verbose,
#     )

#     if hardcoded_result is not None:
#         return hardcoded_result

#     total: int = np.sum(row_sums)
#     n = len(row_sums)

#     if not binary_matrix:  # Symmetric, non-negative matrices
#         if (
#             diagonal_sum is None
#         ):  # Symmetric, non-negative matrices, no diagonal constraint
#             if (
#                 force_second_order
#             ):  # Only case where we might force use of the second order estimate
#                 alpha_dm = _alpha_symmetric_2(
#                     total, n, diagonal_sum=diagonal_sum, alpha=alpha
#                 )
#                 log_count_all = _util._log_binom(
#                     total / 2 + alpha * n * (n + 1) / 2 - 1,
#                     alpha * n * (n + 1) / 2 - 1,
#                 )
#                 log_P = _util._log_P_dirichlet_multinomial(
#                     row_sums, alpha_dm
#                 )
#                 return log_count_all + log_P
#             alpha_plus, alpha_minus = _alpha_symmetric_3(
#                 total, n, diagonal_sum=diagonal_sum, alpha=alpha
#             )
#             log_count_all = _util._log_binom(
#                 total / 2 + alpha * n * (n + 1) / 2 - 1,
#                 alpha * n * (n + 1) / 2 - 1,
#             )
#             log_P_1 = _util._log_P_dirichlet_multinomial(
#                 row_sums, alpha_plus
#             )
#             log_P_2 = _util._log_P_dirichlet_multinomial(
#                 row_sums, alpha_minus
#             )
#             return log_count_all + _util._log_sum_exp([log_P_1, log_P_2]) - float(np.log(2)) # Mean of the probabilities
#         # Symmetric, non-negative matrices, diagonal constraint
#         alpha_dm = _alpha_symmetric_2(
#             total, n, diagonal_sum=diagonal_sum, alpha=alpha
#         )
#         log_count_all = _util._log_binom(diagonal_sum / 2 + alpha * n - 1, alpha * n - 1)
#         log_count_all += _util._log_binom(
#             (total - diagonal_sum) / 2 + alpha * n * (n - 1) / 2 - 1,
#             alpha * n * (n - 1) / 2 - 1,
#         )
#         log_P = _util._log_P_dirichlet_multinomial(
#             row_sums, alpha_dm
#         )
#         return log_count_all + log_P
#     else:  # Binary symmetric matrices
#         if binary_multinomial_estimate:  # Use the multinomial estimate for binary matrices (advantage of never being too far off)
#             result = float(
#                 _util._log_binom(n * (n - 1) / 2, total / 2)
#                 - total * np.log(n)
#                 + _util._log_factorial(total)
#             )
#             for k in row_sums:
#                 result -= _util._log_factorial(k)
#             # Dirichlet-multinomial weight (note that this is a trivial calculation for binary matrices)
#             result += total / 2 * np.log(alpha)
#             return result
#         else: # Binary symmetric matrices, pseudo Dirichlet-Multinomial estimate
#             alpha_dm = _alpha_symmetric_binary(total, n) # Can be negative
#             log_count_all = _util._log_binom(n * (n - 1) / 2, total / 2) + total / 2 * np.log(alpha)
#             log_P = _util._log_P_dirichlet_multinomial(
#                 row_sums, alpha_dm
#             )
#             return log_count_all + log_P
