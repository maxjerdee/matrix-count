import numpy as np
from . import _util
from . import _input_output

######################################
# Symmetric matrices
#######################
# Unbounded

def alpha_symmetric_2(matrix_total, n, diagonal_sum=None, alpha=1.0):
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
    # TODO: Implement this for general alpha
    if diagonal_sum is None:
        numerator = matrix_total + (n + 1) * (matrix_total + n * (matrix_total - 1) - 2) * alpha
        denominator = 2 * (matrix_total - 1) + n * ((n + 1) * alpha + matrix_total - 2)
        result = numerator / denominator
        return result
    result = -(((-1 + n) * diagonal_sum**2 * (2 + (-1 + n) * n * alpha) + 
                2 * (-1 + n) * n * diagonal_sum * alpha * (2 + (-1 + n) * n * alpha) - 
                (-1 + n) * matrix_total**2 * (1 + n * alpha) * (2 + (-1 + n) * n * alpha) + 
                (-2 + n) * (matrix_total - diagonal_sum) * (1 + n * alpha) * (matrix_total - diagonal_sum + (-1 + n) * n * alpha))
              / (n * ((-1 + n) * diagonal_sum**2 * (2 + (-1 + n) * n * alpha) + 
                      2 * (-1 + n) * n * diagonal_sum * alpha * (2 + (-1 + n) * n * alpha) - 
                      (-1 + n) * matrix_total * (1 + n * alpha) * (2 + (-1 + n) * n * alpha) + 
                      (-2 + n) * (matrix_total - diagonal_sum) * (1 + n * alpha) * (matrix_total - diagonal_sum + (-1 + n) * n * alpha))))
    return result

def alpha_symmetric_3(matrix_total, n, diagonal_sum=None, alpha=1.0):
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
        common_numerator = (
            matrix_total**2 * (4 + n + (1 + n) * (4 + n * (11 + 4 * n)) * alpha +
                               n * (1 + n)**3 * (2 + n) * alpha**2) +
            (1 + n) * matrix_total * (-4 + alpha * (-12 + n * (-17 - 11 * n -
            (1 + n) * (7 + n * (4 + 3 * n)) * alpha +
            n * (1 + n)**3 * alpha**2))) +
            (1 + n)**2 * alpha * (8 + n * (4 + alpha * (2 +
            n * (7 + 2 * n - (1 + n) * (4 + n) * alpha)))))
        sqrt_term = np.sqrt(
            (-1 + alpha + n * alpha) *
            (matrix_total - (1 + n) * alpha) *
            (matrix_total + n * (1 + n) * alpha) *
            (-4 - 4 * n + 4 * matrix_total + 3 * n * matrix_total + n * (1 + n) * (n * (-1 + matrix_total) + matrix_total) * alpha) *
            (4 * (-1 + matrix_total) + n * (-4 + 5 * matrix_total + (1 + n) * (-5 + 4 * matrix_total + n * (-4 + 5 * matrix_total)) * alpha +
                                            n * (1 + n)**2 * (-2 + n * (-1 + matrix_total) + matrix_total) * alpha**2))
        )
        denominator = (
            8 * (1 + n)**2 - 2 * (1 + n) * (8 + 5 * n) * matrix_total +
            2 * (4 + n * (5 + 2 * n)) * matrix_total**2 +
            n * (1 + n) * (2 * n * (1 + 2 * n) - 3 * n * (3 + n) * matrix_total +
                           (4 + n * (3 + n)) * matrix_total**2 - 2 * (1 + matrix_total)) * alpha +
            n**2 * (1 + n)**2 * (-3 + n * (-5 + matrix_total) + 5 * matrix_total) * alpha**2 +
            2 * n**3 * (1 + n)**3 * alpha**3
        )

        alpha_plus = (common_numerator + sqrt_term) / denominator
        alpha_minus = (common_numerator - sqrt_term) / denominator

        return alpha_plus, alpha_minus
    raise NotImplementedError("Not yet implemented.")

def estimate_log_symmetric_matrices(row_sums, *, diagonal_sum=None, index_partition=None, block_sums=None, alpha=1.0, estimate_order=3, verbose=False):
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
    _input_output._log_symmetric_matrices_check_arguments(row_sums, diagonal_sum=diagonal_sum, index_partition=index_partition, block_sums=block_sums, alpha=alpha, estimate_order=estimate_order, verbose=verbose)

    # Remove empty margins
    row_sums, diagonal_sum, index_partition, block_sums = _input_output._simplify_input(row_sums, diagonal_sum=diagonal_sum, index_partition=index_partition, block_sums=block_sums)

    # Check for hardcoded cases
    hardcoded_result = _input_output._log_symmetric_matrices_hardcoded(row_sums, diagonal_sum=diagonal_sum, index_partition=index_partition, block_sums=block_sums, alpha=alpha, estimate_order=estimate_order)
    if hardcoded_result is not None:
        return hardcoded_result

    matrix_total = np.sum(row_sums)
    n = len(row_sums)
    if diagonal_sum is None:
        if estimate_order == 2:
            alpha_dm = alpha_symmetric_2(matrix_total, n, diagonal_sum=diagonal_sum, alpha=alpha)
            result = _util._log_binom(matrix_total / 2 + n * (n + 1) / 2 - 1, n * (n + 1) / 2 - 1)
            log_p = -_util._log_binom(matrix_total + n * alpha_dm - 1, n * alpha_dm - 1)
            for k in row_sums:
                log_p += _util._log_binom(k + alpha_dm - 1, alpha_dm - 1)
            result += log_p
            return result
        alpha_plus, alpha_minus = alpha_symmetric_3(matrix_total, n, diagonal_sum=diagonal_sum, alpha=alpha)
        log_1 = _util._log_binom(matrix_total / 2 + n * (n + 1) / 2 - 1, n * (n + 1) / 2 - 1)
        log_1 += -_util._log_binom(matrix_total + n * alpha_plus - 1, n * alpha_plus - 1)
        for k in row_sums:
            log_1 += _util._log_binom(k + alpha_plus - 1, alpha_plus - 1)
        log_2 = _util._log_binom(matrix_total / 2 + n * (n + 1) / 2 - 1, n * (n + 1) / 2 - 1)
        log_2 += -_util._log_binom(matrix_total + n * alpha_minus - 1, n * alpha_minus - 1)
        for k in row_sums:
            log_2 += _util._log_binom(k + alpha_minus - 1, alpha_minus - 1)
        result = _util._log_sum_exp([log_1, log_2]) - np.log(2)
        return result

    if estimate_order == 2:
        alpha_dm = alpha_symmetric_2(matrix_total, n, diagonal_sum=diagonal_sum, alpha=alpha)
        result = _util._log_binom(diagonal_sum / 2 + n - 1, n - 1)
        result += _util._log_binom((matrix_total - diagonal_sum) / 2 + n * (n - 1) / 2 - 1, n * (n - 1) / 2 - 1)
        result += -_util._log_binom(matrix_total + n * alpha_dm - 1, n * alpha_dm - 1)
        for k in row_sums:
            result += _util._log_binom(k + alpha_dm - 1, alpha_dm - 1)
        return result
    raise NotImplementedError("Not yet implemented.")
